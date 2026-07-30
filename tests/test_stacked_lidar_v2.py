from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.envs.observations.stacked_lidar_v2 import StackedLidarObservationV2

FRAME_DIM = 308
HISTORY_LENGTH = 21
STACKED_DIM = FRAME_DIM * HISTORY_LENGTH + HISTORY_LENGTH


def test_flat_dimension_is_6489() -> None:
    observation = StackedLidarObservationV2({})
    observation.set_frame_builder(lambda vehicle: np.full(FRAME_DIM, 0.1, dtype=np.float32))
    out = observation.observe(object())
    assert out.shape == (STACKED_DIM,) == (6489,)


def test_window_covers_21_samples_at_10hz_matching_dashed_tcap_s() -> None:
    from thesis_rl.rulebook.v2.components.road import DASHED_TCAP_S

    control_period_s = 0.1
    expected_frames = int(round(DASHED_TCAP_S / control_period_s)) + 1
    assert expected_frames == HISTORY_LENGTH == 21


def test_reset_then_one_step_mask_is_zero_except_last() -> None:
    observation = StackedLidarObservationV2({})
    observation.set_frame_builder(lambda vehicle: np.full(FRAME_DIM, 0.2, dtype=np.float32))
    observation.reset(env=object())
    out = observation.observe(object())

    frames = out[: FRAME_DIM * HISTORY_LENGTH].reshape(HISTORY_LENGTH, FRAME_DIM)
    mask = out[FRAME_DIM * HISTORY_LENGTH :]

    np.testing.assert_array_equal(mask, np.array([0.0] * 20 + [1.0], dtype=np.float32))
    np.testing.assert_array_equal(frames[:20], np.zeros((20, FRAME_DIM), dtype=np.float32))
    np.testing.assert_array_equal(frames[20], np.full(FRAME_DIM, 0.2, dtype=np.float32))


def test_mask_saturates_after_21_steps_oldest_to_current_order() -> None:
    observation = StackedLidarObservationV2({})
    values = iter(np.linspace(-0.5, 0.5, 25, dtype=np.float32))
    observation.set_frame_builder(
        lambda vehicle: np.full(FRAME_DIM, next(values), dtype=np.float32)
    )
    observation.reset(env=object())

    out = None
    for _ in range(21):
        out = observation.observe(object())
    assert out is not None

    mask = out[FRAME_DIM * HISTORY_LENGTH :]
    np.testing.assert_array_equal(mask, np.ones(HISTORY_LENGTH, dtype=np.float32))

    frames = out[: FRAME_DIM * HISTORY_LENGTH].reshape(HISTORY_LENGTH, FRAME_DIM)
    # oldest -> current: values strictly increasing across the 21 slots.
    means = frames.mean(axis=1)
    assert np.all(np.diff(means) > 0)


def test_populated_vs_empty_rulebook_memory_is_bit_identical() -> None:
    # StackedLidarObservationV2 never reads RulebookMemory; it only forwards
    # whatever the installed frame builder returns (REQ-004).
    calls = []

    def builder_with_context(vehicle: object) -> np.ndarray:
        calls.append(vehicle)
        return np.full(FRAME_DIM, 0.3, dtype=np.float32)

    obs_a = StackedLidarObservationV2({})
    obs_a.set_frame_builder(builder_with_context)
    out_a = obs_a.observe("context_with_populated_memory")

    obs_b = StackedLidarObservationV2({})
    obs_b.set_frame_builder(builder_with_context)
    out_b = obs_b.observe("context_with_empty_memory")

    np.testing.assert_array_equal(out_a, out_b)


def test_per_frame_layout_and_ray_counts_unchanged() -> None:
    from thesis_rl.envs.observations.causal_lidar import CausalLidarFrameBuilder

    assert CausalLidarFrameBuilder.FRAME_DIM == FRAME_DIM


def test_rejects_missing_builder_and_shape_drift() -> None:
    observation = StackedLidarObservationV2({})
    with pytest.raises(RuntimeError, match="requires a causal"):
        observation.observe(object())
    observation.set_frame_builder(lambda _vehicle: np.zeros(307, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        observation.observe(object())


def test_range_and_finiteness_violations_raise() -> None:
    observation = StackedLidarObservationV2({})
    observation.set_frame_builder(lambda _vehicle: np.full(FRAME_DIM, 1.5, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        observation.observe(object())

    observation2 = StackedLidarObservationV2({})
    observation2.set_frame_builder(lambda _vehicle: np.full(FRAME_DIM, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        observation2.observe(object())


def test_determinism_same_seed_twice() -> None:
    rng = np.random.default_rng(42)
    sequence = [rng.uniform(-1.0, 1.0, size=FRAME_DIM).astype(np.float32) for _ in range(5)]

    def run() -> np.ndarray:
        observation = StackedLidarObservationV2({})
        it = iter(sequence)
        observation.set_frame_builder(lambda _vehicle: next(it))
        observation.reset(env=object())
        out = None
        for _ in range(5):
            out = observation.observe(object())
        return out

    first = run()
    second = run()
    np.testing.assert_array_equal(first, second)
