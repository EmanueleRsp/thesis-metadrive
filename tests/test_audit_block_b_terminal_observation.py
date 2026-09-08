"""Audit 2026-09-07, C22: the terminal observation is preprocessed exactly once.

The observation a horizon-truncated transition ends on is the state its value
target bootstraps from, so a transform applied to it twice is not a cosmetic
slip: it corrupts the one state whose estimate feeds back into every earlier
target of the episode. It stayed invisible because the configured preprocessor
is ``identity``, which is idempotent -- the defect is latent, not dormant, and
any affine or stateful preprocessor turns it into a silent numerical error.
"""

from __future__ import annotations

import numpy as np

from thesis_rl.agent.agent import (
    _attach_terminal_observations,
    _preprocess_terminal_observations,
)
from thesis_rl.agent.transition_boundary import normalize_vector_transition_boundary


class _CountingPreprocessor:
    """Deliberately non-idempotent, so a second application cannot hide."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray(observation, dtype=np.float32) * 2.0


def _run_pipeline(
    preprocessor: _CountingPreprocessor,
    *,
    dones: np.ndarray,
    raw_terminals: dict[int, np.ndarray],
    next_obs: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """The three collection-path stages, in the order `Agent.train` runs them."""

    infos: list[dict] = []
    for index in range(len(dones)):
        info: dict = {"TimeLimit.truncated": bool(dones[index])}
        if index in raw_terminals:
            info["terminal_observation"] = raw_terminals[index]
        infos.append(info)

    _preprocess_terminal_observations(infos, preprocessor)
    _terminated, _truncated, final_observations = normalize_vector_transition_boundary(
        dones=dones,
        infos=infos,
        next_observations=next_obs,
    )
    next_processed_obs = np.stack([preprocessor(row) for row in next_obs]).astype(np.float32)
    _attach_terminal_observations(infos, dones, final_observations, next_processed_obs)
    return infos, next_processed_obs


def test_terminal_observation_is_preprocessed_once_not_twice() -> None:
    raw_terminal = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    dones = np.array([True], dtype=bool)
    next_obs = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)

    preprocessor = _CountingPreprocessor()
    infos, next_processed_obs = _run_pipeline(
        preprocessor,
        dones=dones,
        raw_terminals={0: raw_terminal},
        next_obs=next_obs,
    )

    expected = raw_terminal * 2.0
    np.testing.assert_allclose(infos[0]["terminal_observation"], expected)
    np.testing.assert_allclose(infos[0]["final_observation"], expected)
    np.testing.assert_allclose(next_processed_obs[0], expected)
    # One call for the terminal observation, one for the auto-reset row of the
    # `next_obs` batch. A third would be the second application.
    assert preprocessor.calls == 2


def test_every_finished_worker_reports_its_own_terminal_observation() -> None:
    """Two slots finishing on the same step must not share one observation."""

    dones = np.array([True, False, True], dtype=bool)
    raw_terminals = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 3.0], dtype=np.float32),
    }
    next_obs = np.array([[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]], dtype=np.float32)

    preprocessor = _CountingPreprocessor()
    infos, next_processed_obs = _run_pipeline(
        preprocessor,
        dones=dones,
        raw_terminals=raw_terminals,
        next_obs=next_obs,
    )

    np.testing.assert_allclose(next_processed_obs[0], raw_terminals[0] * 2.0)
    np.testing.assert_allclose(next_processed_obs[2], raw_terminals[2] * 2.0)
    # The unfinished slot keeps the live observation, preprocessed once.
    np.testing.assert_allclose(next_processed_obs[1], next_obs[1] * 2.0)
    assert infos[1].get("final_observation") is None
    assert preprocessor.calls == 2 + len(next_obs)


def test_attach_terminal_observations_cannot_reach_a_preprocessor() -> None:
    """The second stage takes no preprocessor, so the defect cannot come back.

    A behavioural test alone would not stop someone re-introducing the call, so
    the absence of the capability is asserted directly: the parameter list is
    the invariant.
    """

    import inspect

    parameters = set(inspect.signature(_attach_terminal_observations).parameters)
    assert "preprocessor" not in parameters
    assert parameters == {"infos", "dones", "final_observations", "next_processed_obs"}
