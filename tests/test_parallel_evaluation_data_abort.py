from __future__ import annotations

import numpy as np

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.execution.deterministic_subproc_vec_env import RuntimeScenarioDataAbort


class _Planner:
    def predict(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.8, -0.8], dtype=np.float32), None


class _EvalEnv:
    """Fixture env that raises a typed data-abort on a chosen reset seed."""

    def __init__(self, abort_seed: int | None = None) -> None:
        self._step = 0
        self._current_seed: int | None = None
        self._abort_seed = abort_seed
        self.quarantined_uids: list[str] = []

    def reset(self, **kwargs):
        self._current_seed = kwargs.get("seed")
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        if self._abort_seed is not None and self._current_seed == self._abort_seed:
            raise RuntimeError("simulated typed data-abort")
        self._step += 1
        done = self._step >= 2
        info = {
            "route_completion": 0.5 if self._step == 1 else 1.0,
            "rule_reward_vector": [-0.2, 0.1] if self._step == 1 else [0.3, -0.1],
            "rule_metadata": {
                "rule_names": ["speed_limit", "goal_progress"],
                "priorities": [1, 0],
                "saturation_ratio_by_rule": {"speed_limit": 0.1, "goal_progress": 0.05},
            },
        }
        if done:
            info.update({"arrive_dest": True, "termination_reason": "success"})
        return np.array([self._step, self._step], dtype=np.float32), 0.1, done, False, info

    def quarantine_scenario_uid(self, scenario_uid: str) -> None:
        self.quarantined_uids.append(str(scenario_uid))


class _LocalVectorSlotProxy:
    def __init__(self, env: _EvalEnv) -> None:
        self.env = env

    def set_rendered_frame(self, frame) -> None:
        _ = frame


class _LocalEvaluationVectorWithAbort:
    num_envs = 2

    def __init__(self, abort_seed: int) -> None:
        self.envs = [_EvalEnv(), _EvalEnv(abort_seed=abort_seed)]
        self.env_method_calls: list[tuple[str, tuple]] = []

    def get_slot_proxy(self, slot: int) -> _LocalVectorSlotProxy:
        return _LocalVectorSlotProxy(self.envs[int(slot)])

    def reset_slots(self, slots, *, seeds=None, options=None):
        _ = options
        return {
            int(slot): self.envs[int(slot)].reset(seed=(seeds or {}).get(int(slot)))
            for slot in slots
        }

    def step_slots(self, actions):
        results = {}
        for slot, action in actions.items():
            env = self.envs[int(slot)]
            try:
                observation, terminated_reward, terminated, truncated, info = env.step(action)
            except RuntimeError:
                results[int(slot)] = RuntimeScenarioDataAbort(
                    slot=int(slot),
                    payload={
                        "reason_code": "INVALID_SIGNAL_TRANSITION",
                        "diagnostics": {
                            "scenario_uid": f"fixture:{env._current_seed}",
                            "environment_step": env._step,
                        },
                        "final_observation": np.array([env._step, env._step], dtype=np.float32),
                    },
                )
                continue
            info = dict(info)
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
            results[int(slot)] = (
                observation,
                terminated_reward,
                terminated or truncated,
                info,
                {},
            )
        return results

    def render_slots(self, render_kwargs=None, *, slots=None):
        _ = render_kwargs
        selected = tuple(sorted(slots or range(self.num_envs)))
        return {slot: np.zeros((2, 2, 3), dtype=np.uint8) for slot in selected}

    def env_method(self, method_name: str, *args, **kwargs):
        self.env_method_calls.append((method_name, args))
        return [getattr(env, method_name)(*args, **kwargs) for env in self.envs]


def _build_agent() -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,)),
    )


def test_parallel_evaluation_excludes_data_abort_episode_from_aggregates() -> None:
    base_seed = 100
    # Episode index 1 is installed into slot 1 with seed base_seed + 1 = 101.
    vector = _LocalEvaluationVectorWithAbort(abort_seed=base_seed + 1)

    metrics = _build_agent().evaluate(
        vector,
        n_eval_episodes=3,
        deterministic=True,
        base_seed=base_seed,
        return_episode_metrics=True,
        show_progress=False,
    )

    coverage = metrics["data_abort_coverage"]
    assert coverage["attempted"] == 3
    assert coverage["invalid"] == 1
    assert coverage["valid"] == 2
    assert coverage["invalid_episodes"] == [
        {
            "episode_idx": 1,
            "scenario_uid": "fixture:101",
            "reason_code": "INVALID_SIGNAL_TRANSITION",
        }
    ]
    # Only the two valid episodes contribute to the aggregate metrics.
    assert metrics["per_episode"]["episode_length"] == [2, 2]
    assert vector.env_method_calls == [("quarantine_scenario_uid", ("fixture:101",))]
