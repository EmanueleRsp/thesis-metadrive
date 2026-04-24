import numpy as np
import pytest

from thesis_rl.adapters.identity import IdentityAdapter


def test_identity_adapter_returns_action_when_shape_matches() -> None:
    adapter = IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,))
    action = adapter(np.array([0.2, -0.3], dtype=np.float32))
    assert np.allclose(action, np.array([0.2, -0.3], dtype=np.float32))


def test_identity_adapter_raises_on_shape_mismatch() -> None:
    adapter = IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,))
    with pytest.raises(ValueError):
        adapter(np.array([0.1, 0.2, 0.3], dtype=np.float32))

