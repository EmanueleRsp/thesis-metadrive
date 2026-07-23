from __future__ import annotations

from thesis_rl.runtime.loops.train_loop import _load_quarantine_state, _save_quarantine_state


class _WorkerEnv:
    def __init__(self, uids: list[str]) -> None:
        self._uids = list(uids)
        self.quarantined: list[str] = []

    def get_quarantined_scenario_uids(self) -> list[str]:
        return sorted(self._uids)

    def quarantine_scenario_uid(self, scenario_uid: str) -> None:
        self.quarantined.append(str(scenario_uid))
        self._uids.append(str(scenario_uid))


class _VecEnv:
    def __init__(self, workers: list[_WorkerEnv]) -> None:
        self.workers = workers

    def env_method(self, method_name: str, *args, **kwargs):
        return [getattr(worker, method_name)(*args, **kwargs) for worker in self.workers]


def test_quarantine_state_round_trips_across_checkpoint_and_resume(tmp_path) -> None:
    path = tmp_path / "latest_quarantine_state.json"
    saving_env = _VecEnv([_WorkerEnv(["waymo:a"]), _WorkerEnv(["waymo:a", "pg:b"])])

    assert _save_quarantine_state(saving_env, path)
    assert path.exists()

    resumed_env = _VecEnv([_WorkerEnv([]), _WorkerEnv([])])
    assert _load_quarantine_state(resumed_env, path)

    for worker in resumed_env.workers:
        assert sorted(worker.quarantined) == ["pg:b", "waymo:a"]


def test_quarantine_load_is_a_no_op_without_a_persisted_file(tmp_path) -> None:
    path = tmp_path / "missing_quarantine_state.json"
    env = _VecEnv([_WorkerEnv([])])

    assert not _load_quarantine_state(env, path)
    assert env.workers[0].quarantined == []
