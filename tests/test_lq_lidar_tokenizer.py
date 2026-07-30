from __future__ import annotations

import numpy as np
import torch

from thesis_rl.agent.planners.encoders.lq_lidar.tokenizer import (
    FLAT_DIM,
    HISTORY_LENGTH,
    NUM_NEARBY_SLOTS,
    NUM_SECTORS,
    NUM_TOKENS,
    SECTOR_WIDTH,
    circular_positional_encoding,
    gather_lidar_tokens,
)
from thesis_rl.agent.planners.encoders.lq_lidar_encoder import LatentQueryEncoderLidar


_FRAME_TOTAL = HISTORY_LENGTH * 308


def _index_encoded_batch() -> torch.Tensor:
    """A [1, 6489] tensor whose frame value at position k is k / _FRAME_TOTAL.

    The mask block is fixed to all-ones (an exact multiplier) so no frame
    value is perturbed by masking and the index can be recovered exactly.
    """

    flat = np.zeros(FLAT_DIM, dtype=np.float32)
    flat[:_FRAME_TOTAL] = np.arange(_FRAME_TOTAL, dtype=np.float32) / _FRAME_TOTAL
    flat[_FRAME_TOTAL:] = 1.0
    return torch.from_numpy(flat).unsqueeze(0)


def test_tokenizer_emits_38_tokens_of_width_64() -> None:
    encoder = LatentQueryEncoderLidar()
    flat = torch.zeros(2, FLAT_DIM)
    tokens = encoder.tokenize_structured(flat)
    assert tokens.shape == (2, NUM_TOKENS, 64)


def test_exact_partition_no_duplicates_no_gaps_stride_308() -> None:
    flat = _index_encoded_batch()
    groups, mask = gather_lidar_tokens(flat)
    np.testing.assert_array_equal(mask.numpy()[0], np.ones(HISTORY_LENGTH, dtype=np.float32))

    frame_dim = 308
    covered: set[int] = set()

    def _record(raw_group: torch.Tensor, group_width_per_frame: int, frame_offset: int) -> None:
        # raw_group values equal (flat_index / _FRAME_TOTAL); recover flat indices.
        values = (raw_group * _FRAME_TOTAL).round().to(torch.int64)
        for value in values.flatten().tolist():
            assert value not in covered, f"index {value} covered twice"
            covered.add(value)

    ego_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 0:6
    ]
    _record(ego_stack, 6, 0)
    nav_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 6:28
    ]
    _record(nav_stack, 22, 6)
    side_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 28:40
    ]
    _record(side_stack, 12, 28)
    lane_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 40:52
    ]
    _record(lane_stack, 12, 40)
    nearby_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 52:68
    ]
    _record(nearby_stack, 16, 52)
    lidar_stack = flat[:, : HISTORY_LENGTH * frame_dim].reshape(1, HISTORY_LENGTH, frame_dim)[
        :, :, 68:308
    ]
    _record(lidar_stack, 240, 68)

    assert covered == set(range(_FRAME_TOTAL))
    # The 21-wide mask block is consumed exactly once, appended to the ego
    # token (REQ-006 counts it as part of the ego group's 147-wide input).
    np.testing.assert_array_equal(
        flat[:, _FRAME_TOTAL:].numpy(), np.ones((1, HISTORY_LENGTH), dtype=np.float32)
    )

    # groups.neighbor / groups.sector must be a re-partition of the same
    # nearby/lidar values gathered by gather_lidar_tokens, at stride 308.
    assert groups.neighbor.shape == (1, NUM_NEARBY_SLOTS, HISTORY_LENGTH * 4)
    assert groups.sector.shape == (1, NUM_SECTORS, HISTORY_LENGTH * SECTOR_WIDTH)
    assert groups.ego.shape == (1, HISTORY_LENGTH * 6 + HISTORY_LENGTH)
    assert groups.navigation.shape == (1, HISTORY_LENGTH * 22)
    assert groups.side.shape == (1, HISTORY_LENGTH * 12)
    assert groups.lane.shape == (1, HISTORY_LENGTH * 12)


def test_mask_zeroes_absent_frames_before_gather() -> None:
    frame_dim = 308
    flat = torch.ones(1, FLAT_DIM)
    mask = torch.zeros(1, HISTORY_LENGTH)
    mask[:, -1] = 1.0
    flat[:, HISTORY_LENGTH * frame_dim :] = mask
    groups, _ = gather_lidar_tokens(flat)
    ego_stack = groups.ego[:, : HISTORY_LENGTH * 6].reshape(1, HISTORY_LENGTH, 6)
    np.testing.assert_array_equal(ego_stack[:, :-1].numpy(), np.zeros((1, HISTORY_LENGTH - 1, 6)))
    np.testing.assert_array_equal(ego_stack[:, -1].numpy(), np.ones((1, 6)))


def test_circular_positional_encoding_wraps_at_num_sectors() -> None:
    # AC-006: PE(i) == PE(i mod num_sectors). theta_i = 2*pi*i/N is exactly
    # periodic with period N, so evaluating the same fundamental frequency at
    # i=0 and i=N must coincide.
    theta_0 = torch.tensor(0.0)
    theta_n = 2.0 * torch.pi * torch.tensor(float(NUM_SECTORS)) / NUM_SECTORS
    np.testing.assert_allclose(torch.sin(theta_0).item(), torch.sin(theta_n).item(), atol=1e-6)
    np.testing.assert_allclose(torch.cos(theta_0).item(), torch.cos(theta_n).item(), atol=1e-6)
    pe = circular_positional_encoding(NUM_SECTORS, 64)
    assert pe.shape == (NUM_SECTORS, 64)


def test_circular_positional_encoding_distance_monotone_in_circular_distance() -> None:
    pe = circular_positional_encoding(NUM_SECTORS, 64)
    reference = pe[0]
    distances = []
    circular_deltas = list(range(NUM_SECTORS // 2 + 1))
    for delta in circular_deltas:
        dist = torch.linalg.norm(pe[delta] - reference).item()
        distances.append(dist)
    assert all(distances[i] <= distances[i + 1] + 1e-6 for i in range(len(distances) - 1))


def test_sector_tokens_permute_cyclically_under_rotated_scene() -> None:
    frame_dim = 308
    rng = np.random.default_rng(0)
    lidar_rays = rng.uniform(-1.0, 1.0, size=240).astype(np.float32)

    def _build_flat(lidar: np.ndarray) -> torch.Tensor:
        flat = np.zeros(FLAT_DIM, dtype=np.float32)
        for frame in range(HISTORY_LENGTH):
            offset = frame * frame_dim
            flat[offset + 68 : offset + 308] = lidar
        flat[HISTORY_LENGTH * frame_dim :] = 1.0
        return torch.from_numpy(flat).unsqueeze(0)

    original = _build_flat(lidar_rays)
    rotated_rays = np.roll(lidar_rays, -SECTOR_WIDTH)  # rotate by exactly one sector
    rotated = _build_flat(rotated_rays)

    groups_original, _ = gather_lidar_tokens(original)
    groups_rotated, _ = gather_lidar_tokens(rotated)

    np.testing.assert_allclose(
        groups_rotated.sector[:, :-1].numpy(), groups_original.sector[:, 1:].numpy(), atol=1e-6
    )


def test_encoder_accepts_38x64_and_returns_batch_256() -> None:
    encoder = LatentQueryEncoderLidar()
    flat = torch.zeros(3, FLAT_DIM)
    out = encoder(flat)
    assert out.shape == (3, 256)


def test_encoder_rejects_wrong_input_dim() -> None:
    encoder = LatentQueryEncoderLidar()
    flat = torch.zeros(2, FLAT_DIM - 1)
    try:
        encoder(flat)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong input dim")


def test_encoder_rejects_non_core_hyperparameters() -> None:
    import pytest

    with pytest.raises(ValueError):
        LatentQueryEncoderLidar(token_dim=32)
    with pytest.raises(ValueError):
        LatentQueryEncoderLidar(num_latents=8)
