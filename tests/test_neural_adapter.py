from __future__ import annotations

from pathlib import Path

import numpy as np

from thesis_rl.adapters.neural_adapter import NeuralAdapter


def test_neural_adapter_call_shape_and_clip() -> None:
    adapter = NeuralAdapter(expected_shape=(2,), clip=True, low=-1.0, high=1.0, device="cpu")

    out = adapter(np.array([2.0, -3.0], dtype=np.float32))
    assert out.shape == (2,)
    assert out.dtype == np.float32
    assert np.all(out <= 1.0)
    assert np.all(out >= -1.0)


def test_neural_adapter_training_step_updates_loss() -> None:
    adapter = NeuralAdapter(expected_shape=(2,), batch_size=4, update_interval=1, device="cpu")
    adapter.begin_training()

    for _ in range(8):
        _ = adapter(np.array([0.5, -0.2], dtype=np.float32))
        adapter.maybe_update()

    assert np.isfinite(adapter.last_loss)
    adapter.end_training()


def test_neural_adapter_save_load(tmp_path: Path) -> None:
    adapter = NeuralAdapter(expected_shape=(2,), batch_size=2, update_interval=1, device="cpu")
    adapter.begin_training()

    for _ in range(4):
        _ = adapter(np.array([0.1, -0.1], dtype=np.float32))
        adapter.maybe_update()

    checkpoint = tmp_path / "neural_adapter.pt"
    adapter.save(str(checkpoint))
    assert checkpoint.exists()

    loaded = NeuralAdapter(expected_shape=(2,), device="cpu")
    loaded.load(str(checkpoint))
    out = loaded(np.array([0.2, -0.2], dtype=np.float32))
    assert out.shape == (2,)
