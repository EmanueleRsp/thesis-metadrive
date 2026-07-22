from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from thesis_rl.envs import factory
from thesis_rl.runtime.wiring import builders


def _cfg(*, eval_workers: int = 12, test_workers: int = 8, start_method: str = "spawn"):
    return OmegaConf.create(
        {
            "experiment": {
                "eval_workers": eval_workers,
                "test_workers": test_workers,
                "evaluation_start_method": start_method,
            },
            "env": {"name": "metadrive", "config": {"start_seed": 0, "num_scenarios": 1}},
        }
    )


def test_evaluation_worker_defaults_are_independent_from_training() -> None:
    cfg = _cfg()
    assert builders.evaluation_num_workers(cfg, final=False) == 12
    assert builders.evaluation_num_workers(cfg, final=True) == 8


def test_evaluation_worker_count_rejects_non_positive_values() -> None:
    cfg = _cfg(eval_workers=0)
    try:
        builders.evaluation_num_workers(cfg, final=False)
    except ValueError as exc:
        assert "eval_workers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-positive evaluation workers must fail")


def test_single_evaluation_worker_keeps_sequential_builder(monkeypatch) -> None:
    cfg = _cfg()
    sentinel = object()
    monkeypatch.setattr(builders, "build_env", lambda _cfg, overrides: sentinel)

    result = builders.build_eval_env(
        cfg,
        {"start_seed": 20},
        n_eval_episodes=10,
        workers=1,
    )

    assert result is sentinel


def test_parallel_evaluation_requires_spawn() -> None:
    cfg = _cfg(start_method="fork")
    try:
        builders.build_eval_env(cfg, None, n_eval_episodes=2, workers=2)
    except ValueError as exc:
        assert "spawn" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("parallel evaluation must require spawn")


def test_parallel_evaluation_caps_workers_to_episode_count(monkeypatch) -> None:
    cfg = _cfg()
    captured: dict[str, object] = {}

    class _FakeVector:
        def __init__(self, env_fns, **kwargs) -> None:
            captured["worker_count"] = len(env_fns)
            captured["kwargs"] = kwargs

    monkeypatch.setattr(builders, "DeterministicSubprocVecEnv", _FakeVector)
    result = builders.build_eval_env(cfg, None, n_eval_episodes=2, workers=12)

    assert isinstance(result, _FakeVector)
    assert captured["worker_count"] == 2
    assert captured["kwargs"] == {"start_method": "spawn", "auto_reset": False}


def test_parallel_evaluation_forwards_explicit_worker_thread_limits(monkeypatch) -> None:
    cfg = _cfg()
    cfg.env.vectorized = {
        "worker_num_threads": 2,
        "worker_library_num_threads": 3,
    }
    captured: dict[str, object] = {}

    class _FakeVector:
        def __init__(self, env_fns, **kwargs) -> None:
            captured["worker_count"] = len(env_fns)
            captured["kwargs"] = kwargs

    monkeypatch.setattr(builders, "DeterministicSubprocVecEnv", _FakeVector)
    result = builders.build_eval_env(cfg, None, n_eval_episodes=2, workers=12)

    assert isinstance(result, _FakeVector)
    assert captured["worker_count"] == 2
    assert captured["kwargs"] == {
        "start_method": "spawn",
        "auto_reset": False,
        "torch_num_threads": 2,
        "numeric_library_num_threads": 3,
    }


def test_scenarionet_sequence_helper_honors_acl_schedule(monkeypatch) -> None:
    records = tuple(
        SimpleNamespace(
            validation_status="valid",
            rulebook_eligible=True,
            scenario_uid=f"uid-{index}",
            split="test",
            runtime_index=index,
            primary_arm=arm,
            source="waymo",
        )
        for index, arm in enumerate(("A1_traffic", "A2_junction") * 3)
    )
    catalog = SimpleNamespace(valid_records=lambda split=None: records)
    monkeypatch.setattr("thesis_rl.scenarios.catalog.read_scenario_catalog", lambda _path: catalog)
    cfg_env = OmegaConf.create(
        {
            "split": "test",
            "catalog_path": "catalog.parquet",
            "global_seed": 7,
            "provider": {
                "kind": "uniform",
                "strict": True,
                "allow_fallback": False,
                "source_probability": {"waymo": 1.0, "pg": 0.0},
            },
            "config": {"start_scenario_index": 0, "num_scenarios": -1},
        }
    )
    schedule = ("A1_traffic", "A2_junction", "A1_traffic", "A2_junction")

    indices = factory.scenario_evaluation_runtime_indices(
        cfg_env,
        len(schedule),
        arm_schedule=schedule,
        source_schedule=("waymo",) * len(schedule),
    )

    by_index = {record.runtime_index: record for record in records}
    assert [by_index[index].primary_arm for index in indices] == list(schedule)
    assert [by_index[index].source for index in indices] == ["waymo"] * len(schedule)
