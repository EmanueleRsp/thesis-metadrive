from __future__ import annotations

import pytest

from thesis_rl.envs.factory import _configure_agent_observation


def test_semantic_observation_rejects_future_trajectory_flag() -> None:
    with pytest.raises(ValueError, match="future time-indexed"):
        _configure_agent_observation(
            {},
            {
                "type": "semantic_state",
                "expose_time_indexed_future_trajectory": True,
            },
        )


def test_semantic_observation_rejects_future_signal_phase_flag() -> None:
    with pytest.raises(ValueError, match="future signal"):
        _configure_agent_observation(
            {},
            {"type": "semantic_state", "expose_future_signal_phase": True},
        )
