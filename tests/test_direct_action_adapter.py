import numpy as np

from thesis_rl.adapters.direct_action import DirectActionAdapter


def test_direct_action_adapter_clips_bounds() -> None:
    adapter = DirectActionAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,))
    action = adapter(np.array([2.0, -2.0], dtype=np.float32))
    assert np.allclose(action, np.array([1.0, -1.0], dtype=np.float32))
