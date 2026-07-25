"""Regression tests for EVAL-PROTOCOL v1.0 REQ-008.

REQ-008 requires that R1--R3 constraint macro-rule violation rates and
episode-minimum margins be aggregated only over steps the authoritative
Rulebook v2 result marks applicable for that rule, first per episode, then
averaged across episodes with at least one applicable step; episodes with
zero applicable steps for a rule are excluded from that rule's aggregate
(not counted as zero violations), and the exclusion count is reported.

See `docs/specifications/evaluation_protocol_v1.0_specification.md` REQ-008
(§6) and §7.2, and `docs/implementation/evaluation_protocol_v1.0_exec_plan.md`
Milestone 10.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor

RULE_NAME = "test_rule"


class _Planner:
    def predict(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.5, -0.5], dtype=np.float32), None


def _build_agent() -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,)),
    )


class _ApplicabilityEnv:
    """Three fixed 2-step episodes exercising REQ-008's exclusion rule.

    Episode 0: rule applicable at both steps, one violated -> episode
        violation rate 0.5.
    Episode 1: rule NOT applicable at either step (margin still emitted as
        0.0, mirroring `aggregate_max_component`'s NOT_APPLICABLE cost=0.0)
        -> zero applicable steps, must be excluded from the rule's aggregate.
    Episode 2: rule applicable at both steps, none violated -> episode
        violation rate 0.0.
    """

    EPISODES: tuple[tuple[tuple[bool, float], ...], ...] = (
        ((True, -0.1), (True, 0.2)),
        ((False, 0.0), (False, 0.0)),
        ((True, 0.3), (True, 0.4)),
    )

    def __init__(self, *, fixed_pattern: tuple[tuple[bool, float], ...] | None = None) -> None:
        # `fixed_pattern` lets a single-episode-per-slot parallel-evaluation
        # env instance replay one specific pattern regardless of how many
        # times `reset()` fires; without it, the instance cycles through
        # `EPISODES` in order across resets, matching the serial evaluation
        # loop's episode-index progression on one reused env instance.
        self._fixed_pattern = fixed_pattern
        self._episode_idx = -1
        self._step = 0

    def reset(self, **kwargs):
        _ = kwargs
        self._episode_idx += 1
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        pattern = (
            self._fixed_pattern
            if self._fixed_pattern is not None
            else self.EPISODES[self._episode_idx]
        )
        applicable, margin = pattern[self._step]
        self._step += 1
        done = self._step >= len(pattern)
        info = {
            "rule_reward_vector": [margin],
            "rule_metadata": {
                "rule_names": [RULE_NAME],
                "priorities": [0],
            },
            "rule_components": {
                RULE_NAME: {
                    "applicable": applicable,
                    "status": "violated" if margin < 0 else "satisfied",
                },
            },
        }
        if done:
            info.update({"arrive_dest": True, "termination_reason": "success"})
        obs = np.array([self._step, self._step], dtype=np.float32)
        return obs, 0.1, done, False, info


def _rule_row(metrics: dict) -> dict:
    rows = [row for row in metrics["per_rule"] if row["rule_name"] == RULE_NAME]
    assert len(rows) == 1, metrics["per_rule"]
    return rows[0]


def test_req008_violation_rate_excludes_episodes_with_no_applicable_steps() -> None:
    metrics = _build_agent().evaluate(
        _ApplicabilityEnv(),
        n_eval_episodes=3,
        deterministic=True,
        show_progress=False,
    )
    row = _rule_row(metrics)

    # Only episodes 0 and 2 have >=1 applicable step; episode 1 is excluded.
    assert row["applicable_episode_count"] == 2
    assert row["excluded_episode_count"] == 1
    # RuleViolationRate = mean(EpisodeViolationRate) over included episodes
    # only: mean([1/2, 0/2]) = 0.25, not the naive 1/3 an episode-count-based
    # (pre-REQ-008) denominator would have produced.
    assert row["violation_rate"] == 0.25


def test_req008_min_margin_uses_applicable_steps_only() -> None:
    metrics = _build_agent().evaluate(
        _ApplicabilityEnv(),
        n_eval_episodes=3,
        deterministic=True,
        show_progress=False,
    )
    row = _rule_row(metrics)

    # Episode-minimum margins over applicable steps only: episode 0 -> -0.1,
    # episode 2 -> 0.3; episode 1 contributes no entry (zero applicable
    # steps), so it must not pull the aggregate toward its all-zero margins.
    assert row["min_margin"] == -0.1
    assert row["max_margin"] == 0.3
    assert row["mean_margin"] == pytest.approx(0.1)


def test_req008_zero_applicable_steps_never_counted_as_zero_violation() -> None:
    """A rule inapplicable for an entire episode must not silently emit a
    misleading zero-violation contribution for that episode (fail closed:
    exclude, don't zero-fill)."""
    metrics = _build_agent().evaluate(
        _ApplicabilityEnv(),
        n_eval_episodes=3,
        deterministic=True,
        show_progress=False,
    )
    row = _rule_row(metrics)

    naive_denominator_rate = 1.0 / 3.0
    assert row["violation_rate"] != naive_denominator_rate
    assert row["excluded_episode_count"] == 1


def test_req008_serial_and_parallel_paths_agree() -> None:
    """`_ParallelEvaluationEpisode`/`_aggregate_parallel_evaluation` and the
    serial `Agent.evaluate()` loop duplicate REQ-008's aggregation
    independently; they must produce identical per-rule rows."""

    class _Slot:
        def __init__(self, env: _ApplicabilityEnv) -> None:
            self.env = env

        def set_rendered_frame(self, frame) -> None:
            _ = frame

    class _Vector:
        num_envs = 3

        def __init__(self) -> None:
            # One fixed episode pattern per slot (each slot's env instance
            # runs exactly one episode for this 3-episode/3-worker case), so
            # this must reproduce the same three episode patterns the serial
            # `_ApplicabilityEnv` instance cycles through across its resets.
            self.envs = [
                _ApplicabilityEnv(fixed_pattern=pattern) for pattern in _ApplicabilityEnv.EPISODES
            ]

        def get_slot_proxy(self, slot: int) -> _Slot:
            return _Slot(self.envs[int(slot)])

        def reset_slots(self, slots, *, seeds=None, options=None):
            _ = (seeds, options)
            return {int(slot): self.envs[int(slot)].reset() for slot in slots}

        def step_slots(self, actions):
            results = {}
            for slot, action in actions.items():
                observation, reward, done, truncated, info = self.envs[int(slot)].step(action)
                info = dict(info)
                info["terminated"] = bool(done)
                info["truncated"] = bool(truncated)
                results[int(slot)] = (observation, reward, done or truncated, info, {})
            return results

        def render_slots(self, render_kwargs=None, *, slots=None):
            _ = render_kwargs
            selected = tuple(sorted(slots or range(self.num_envs)))
            return {slot: np.zeros((2, 2, 3), dtype=np.uint8) for slot in selected}

    sequential = _build_agent().evaluate(
        _ApplicabilityEnv(), n_eval_episodes=3, deterministic=True, show_progress=False
    )
    parallel = _build_agent().evaluate(
        _Vector(), n_eval_episodes=3, deterministic=True, show_progress=False
    )

    assert _rule_row(parallel) == _rule_row(sequential)
