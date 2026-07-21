from __future__ import annotations

import numpy as np

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor


class _Planner:
    def predict(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.8, -0.8], dtype=np.float32), None


class _EvalEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self, **kwargs):
        _ = kwargs
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
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


class _LocalVectorSlotProxy:
    def __init__(self, env: _EvalEnv) -> None:
        self.env = env

    def set_rendered_frame(self, frame) -> None:
        _ = frame


class _LocalEvaluationVector:
    num_envs = 2

    def __init__(self) -> None:
        self.envs = [_EvalEnv(), _EvalEnv()]
        self.render_calls: list[tuple[int, ...]] = []

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
            observation, reward, terminated, truncated, info = self.envs[int(slot)].step(action)
            info = dict(info)
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
            results[int(slot)] = (observation, reward, terminated or truncated, info, {})
        return results

    def render_slots(self, render_kwargs=None, *, slots=None):
        _ = render_kwargs
        selected = tuple(sorted(slots or range(self.num_envs)))
        self.render_calls.append(selected)
        return {slot: np.zeros((2, 2, 3), dtype=np.uint8) for slot in selected}


def _build_agent() -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,)),
    )


def test_parallel_evaluation_preserves_sequential_metrics() -> None:
    sequential = _build_agent().evaluate(
        _EvalEnv(),
        n_eval_episodes=4,
        deterministic=True,
        base_seed=100,
        return_episode_metrics=True,
        show_progress=False,
    )
    parallel = _build_agent().evaluate(
        _LocalEvaluationVector(),
        n_eval_episodes=4,
        deterministic=True,
        base_seed=100,
        return_episode_metrics=True,
        show_progress=False,
    )

    for key in (
        "mean_reward",
        "std_reward",
        "mean_env_reward",
        "collision_rate",
        "out_of_road_rate",
        "success_rate",
        "route_completion",
        "top_rule_violation_rate",
        "avg_error_value",
        "max_error_value",
        "counterexample_rate",
        "violated_rules_ratio",
    ):
        assert parallel[key] == sequential[key]
    assert parallel["per_rule"] == sequential["per_rule"]
    np.testing.assert_array_equal(
        parallel["per_episode"]["returns"], sequential["per_episode"]["returns"]
    )
    assert parallel["per_episode"]["episode_length"] == sequential["per_episode"]["episode_length"]


class _ArtifactRecorder:
    def __init__(self, episode_idx: int) -> None:
        self.episode_idx = episode_idx
        self.steps = 0

    def record_step(self, **kwargs) -> None:
        _ = kwargs
        self.steps += 1

    def finalize_episode(self, *, episode_metrics):
        _ = episode_metrics
        return {
            "video_path": f"episode_{self.episode_idx}.gif",
            "video_authoritative_path": f"episode_{self.episode_idx}.gif",
            "video_manifest_path": None,
            "trajectory_log_path": None,
            "video_recorded_live": True,
            "replay_warning": None,
        }


def test_parallel_evaluation_records_live_artifacts_per_episode() -> None:
    vector = _LocalEvaluationVector()
    metrics = _build_agent().evaluate(
        vector,
        n_eval_episodes=3,
        deterministic=True,
        base_seed=100,
        return_episode_metrics=True,
        show_progress=False,
        artifact_recorder_factory=lambda context: _ArtifactRecorder(int(context["episode_idx"])),
    )

    assert metrics["per_episode"]["video_recorded_live"] == [True, True, True]
    assert metrics["per_episode"]["video_path"] == [
        "episode_0.gif",
        "episode_1.gif",
        "episode_2.gif",
    ]
    assert vector.render_calls
    assert all(call for call in vector.render_calls)
