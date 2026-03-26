import numpy as np
import pytest

from thesis_rl.adapters.direct_action import DirectActionAdapter
from thesis_rl.agents.agent import Agent
from thesis_rl.preprocessors.identity import IdentityPreprocessor


class _DummyLifecycle:
    def __init__(self) -> None:
        self.begin_called = False
        self.end_called = False
        self.steps = 0

    def begin_training(self, total_timesteps: int | None = None) -> None:
        _ = total_timesteps
        self.begin_called = True

    def act(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.2, 0.1], dtype=np.float32)

    def observe_transition(self, observation, action, reward, done, next_observation) -> None:
        _ = (observation, action, reward, done, next_observation)
        self.steps += 1

    def maybe_update(self) -> None:
        return None

    def end_training(self) -> None:
        self.end_called = True


class _DummyPlanner:
    def __init__(self) -> None:
        self.lifecycle = _DummyLifecycle()
        self.saved_path = None

    def predict(self, observation, deterministic: bool = False):
        # Fixed out-of-bounds action to verify adapter clipping is applied.
        return np.array([2.0, -2.0], dtype=np.float32), None

    def get_lifecycle(self):
        return self.lifecycle

    def save(self, checkpoint_path) -> None:
        self.saved_path = checkpoint_path
        return None


class _TrainableDummyAdapter:
    requires_training = True

    def __init__(self) -> None:
        self.saved_path = None
        self.loaded_path = None

    def __call__(self, planner_output):
        return np.asarray(planner_output, dtype=np.float32)

    def begin_training(self) -> None:
        return None

    def maybe_update(self) -> None:
        return None

    def end_training(self) -> None:
        return None

    def save(self, checkpoint_path: str) -> None:
        self.saved_path = checkpoint_path

    def load(self, checkpoint_path: str) -> None:
        self.loaded_path = checkpoint_path


class _StatelessDummyAdapter(_TrainableDummyAdapter):
    requires_training = False


class _DummyEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self):
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step += 1
        obs = np.array([self._step, self._step], dtype=np.float32)
        reward = 0.1
        done = self._step >= 2
        truncated = False
        return obs, reward, done, truncated, {}


class _DummyEnvWithRuleMetadata(_DummyEnv):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["rule_metadata"] = {
            "saturation_ratio_by_rule": {
                "speed_limit": 0.25,
                "goal_progress": 0.10,
            }
        }
        return obs, reward, done, truncated, info


class _DummyEnvWithCurriculumSignals(_DummyEnv):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        if self._step == 1:
            info.update(
                {
                    "route_completion": 0.5,
                    "rule_reward_vector": [-0.2, 0.1],
                    "rule_metadata": {
                        "saturation_ratio_by_rule": {
                            "speed_limit": 0.1,
                            "goal_progress": 0.05,
                        }
                    },
                }
            )
        else:
            info.update(
                {
                    "arrive_dest": True,
                    "route_completion": 1.0,
                    "out_of_road": False,
                    "crash_vehicle": False,
                    "rule_reward_vector": [0.3, -0.1],
                    "rule_metadata": {
                        "saturation_ratio_by_rule": {
                            "speed_limit": 0.2,
                            "goal_progress": 0.1,
                        }
                    },
                }
            )
        return obs, reward, done, truncated, info


def test_agent_predict_uses_pipeline_order() -> None:
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    planner = _DummyPlanner()
    adapter = DirectActionAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,))
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    obs = np.array([1.0, 2.0], dtype=np.float64)
    action, _ = agent.predict(obs)

    assert action.dtype == np.float32
    assert np.allclose(action, np.array([1.0, -1.0], dtype=np.float32))


def test_agent_train_uses_lifecycle_only() -> None:
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    planner = _DummyPlanner()
    adapter = DirectActionAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,))
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)
    env = _DummyEnv()

    agent.train(env=env, total_timesteps=3, deterministic=False, log_interval=0)

    assert planner.lifecycle.begin_called is True
    assert planner.lifecycle.end_called is True
    assert planner.lifecycle.steps == 3


def test_agent_save_saves_trainable_adapter(tmp_path) -> None:
    planner = _DummyPlanner()
    adapter = _TrainableDummyAdapter()
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    ckpt = tmp_path / "baseline_td3"
    agent.save(ckpt)

    expected_adapter_ckpt = tmp_path / "baseline_td3.adapter.pt"
    assert planner.saved_path == ckpt
    assert adapter.saved_path == str(expected_adapter_ckpt)


def test_agent_load_adapter_strict_for_trainable(tmp_path) -> None:
    planner = _DummyPlanner()
    adapter = _TrainableDummyAdapter()
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    with pytest.raises(FileNotFoundError):
        agent.load_adapter(tmp_path / "missing_td3.zip", strict=True)


def test_agent_load_adapter_noop_for_stateless(tmp_path) -> None:
    planner = _DummyPlanner()
    adapter = _StatelessDummyAdapter()
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    # Should not raise even if no adapter checkpoint exists.
    agent.load_adapter(tmp_path / "missing_td3.zip", strict=True)
    assert adapter.loaded_path is None


def test_agent_evaluate_reports_rule_saturation_metric() -> None:
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    planner = _DummyPlanner()
    adapter = DirectActionAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,))
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    metrics = agent.evaluate(_DummyEnvWithRuleMetadata(), n_eval_episodes=2, deterministic=True)

    assert "mean_reward" in metrics
    assert "std_reward" in metrics
    assert "mean_rule_saturation_max" in metrics
    assert "collision_rate" in metrics
    assert "out_of_road_rate" in metrics
    assert "success_rate" in metrics
    assert "route_completion" in metrics
    assert "top_rule_violation_rate" in metrics
    assert metrics["mean_rule_saturation_max"] == 0.25


def test_agent_evaluate_reports_curriculum_metrics() -> None:
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    planner = _DummyPlanner()
    adapter = DirectActionAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,))
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    metrics = agent.evaluate(_DummyEnvWithCurriculumSignals(), n_eval_episodes=2, deterministic=True)

    assert metrics["collision_rate"] == 0.0
    assert metrics["out_of_road_rate"] == 0.0
    assert metrics["success_rate"] == 1.0
    assert metrics["route_completion"] == 1.0
    assert metrics["top_rule_violation_rate"] == 0.5
