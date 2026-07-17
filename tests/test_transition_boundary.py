import numpy as np
import pytest

from thesis_rl.agent.transition_boundary import normalize_vector_transition_boundary


def test_vector_boundary_preserves_separate_flags_and_final_observation() -> None:
    infos = [{"final_observation": np.asarray([9.0], dtype=np.float32)}, {}]
    terminated, truncated, final_obs = normalize_vector_transition_boundary(
        dones=np.asarray([True, False]),
        infos=infos,
        next_observations=np.asarray([[0.0], [2.0]], dtype=np.float32),
        terminated=np.asarray([True, False]),
        truncated=np.asarray([False, False]),
    )

    np.testing.assert_array_equal(terminated, [True, False])
    np.testing.assert_array_equal(truncated, [False, False])
    np.testing.assert_array_equal(final_obs, [[9.0], [2.0]])
    assert infos[0]["terminal_observation"] is infos[0]["final_observation"]


def test_vector_boundary_timeout_requires_final_observation() -> None:
    with pytest.raises(ValueError, match="final_observation"):
        normalize_vector_transition_boundary(
            dones=np.asarray([True]),
            infos=[{}],
            next_observations=np.asarray([[0.0]], dtype=np.float32),
            terminated=np.asarray([False]),
            truncated=np.asarray([True]),
        )


def test_vector_boundary_termination_prevails_when_both_flags_are_true() -> None:
    infos = [{"terminal_observation": np.asarray([4.0], dtype=np.float32)}]
    terminated, truncated, _ = normalize_vector_transition_boundary(
        dones=np.asarray([True]),
        infos=infos,
        next_observations=np.asarray([[0.0]], dtype=np.float32),
        terminated=np.asarray([True]),
        truncated=np.asarray([True]),
    )

    assert bool(terminated[0])
    assert bool(truncated[0])
    assert infos[0]["TimeLimit.truncated"] is False
