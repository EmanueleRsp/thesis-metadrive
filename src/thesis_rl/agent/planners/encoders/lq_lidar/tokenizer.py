"""Strided gather over the 6489-wide stacked_lidar_v2 observation.

Layout: `frame_dim=308` per timestep, 21 timesteps oldest-to-current, followed
by a 21-wide validity mask (`src/thesis_rl/envs/observations/
stacked_lidar_v2.py`). Per-frame group offsets are the frozen
`CausalLidarFrameBuilder` layout: ``ego(6) | navigation(22) | side(12) |
lane(12) | nearby(16) | lidar(240)``
(`src/thesis_rl/envs/observations/causal_lidar.py`).

The tokenizer partitions the 6489 values into 38 groups with no overlap and no
gap (verified by `TEST-008` against an index-encoded fixture). Neighbor and
LiDAR-sector groups deliberately use one shared gather routine applied at
different slot offsets, with no per-slot identity embedding added downstream
(`DEC-008`, mirroring the `ADR-026` precedent).
"""

from __future__ import annotations

import torch

HISTORY_LENGTH = 21
FRAME_DIM = 308
FLAT_DIM = FRAME_DIM * HISTORY_LENGTH + HISTORY_LENGTH  # 6489

_EGO_OFFSET, _EGO_WIDTH = 0, 6
_NAV_OFFSET, _NAV_WIDTH = 6, 22
_SIDE_OFFSET, _SIDE_WIDTH = 28, 12
_LANE_OFFSET, _LANE_WIDTH = 40, 12
_NEARBY_OFFSET, _NEARBY_WIDTH = 52, 16
_LIDAR_OFFSET, _LIDAR_WIDTH = 68, 240

NUM_NEARBY_SLOTS = 4
NEARBY_SLOT_WIDTH = _NEARBY_WIDTH // NUM_NEARBY_SLOTS  # 4

NUM_SECTORS = 30
SECTOR_WIDTH = _LIDAR_WIDTH // NUM_SECTORS  # 8

NUM_TOKENS = 1 + 1 + 1 + 1 + NUM_NEARBY_SLOTS + NUM_SECTORS  # 38

if _LIDAR_WIDTH % SECTOR_WIDTH != 0:
    raise ValueError("Causal LiDAR ray count must be divisible by the STECA sector width")
if _NEARBY_WIDTH % NUM_NEARBY_SLOTS != 0:
    raise ValueError("Causal LiDAR nearby block must divide evenly across neighbor slots")


class LidarTokenizerGroups:
    """Container for the 38 pre-projection token inputs, mask-applied."""

    __slots__ = ("ego", "navigation", "side", "lane", "neighbor", "sector")

    def __init__(
        self,
        ego: torch.Tensor,
        navigation: torch.Tensor,
        side: torch.Tensor,
        lane: torch.Tensor,
        neighbor: torch.Tensor,
        sector: torch.Tensor,
    ) -> None:
        self.ego = ego
        self.navigation = navigation
        self.side = side
        self.lane = lane
        self.neighbor = neighbor
        self.sector = sector


def _split_frames_and_mask(flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if flat_obs.ndim != 2 or flat_obs.shape[-1] != FLAT_DIM:
        raise ValueError(
            f"stacked_lidar_v2 tensor must be batch-first [B, {FLAT_DIM}]; got {tuple(flat_obs.shape)}"
        )
    batch_size = flat_obs.shape[0]
    frames = flat_obs[:, : FRAME_DIM * HISTORY_LENGTH].reshape(
        batch_size, HISTORY_LENGTH, FRAME_DIM
    )
    mask = flat_obs[:, FRAME_DIM * HISTORY_LENGTH :]
    return frames, mask


def gather_lidar_tokens(flat_obs: torch.Tensor) -> tuple[LidarTokenizerGroups, torch.Tensor]:
    """Gather every group at stride `FRAME_DIM`, mask-zero absent frames.

    Returns the per-group raw (unprojected) stacked slices and the `[B, 21]`
    validity mask, unchanged, for the caller to append to the ego token.
    """

    frames, mask = _split_frames_and_mask(flat_obs)
    batch_size = frames.shape[0]
    mask_expanded = mask.unsqueeze(-1)  # [B, 21, 1]

    def _group(offset: int, width: int) -> torch.Tensor:
        return frames[:, :, offset : offset + width] * mask_expanded

    ego_stack = _group(_EGO_OFFSET, _EGO_WIDTH).reshape(batch_size, HISTORY_LENGTH * _EGO_WIDTH)
    ego = torch.cat((ego_stack, mask), dim=-1)  # [B, 147]

    navigation = _group(_NAV_OFFSET, _NAV_WIDTH).reshape(batch_size, HISTORY_LENGTH * _NAV_WIDTH)
    side = _group(_SIDE_OFFSET, _SIDE_WIDTH).reshape(batch_size, HISTORY_LENGTH * _SIDE_WIDTH)
    lane = _group(_LANE_OFFSET, _LANE_WIDTH).reshape(batch_size, HISTORY_LENGTH * _LANE_WIDTH)

    nearby_full = _group(_NEARBY_OFFSET, _NEARBY_WIDTH)  # [B, 21, 16]
    nearby_full = nearby_full.reshape(
        batch_size, HISTORY_LENGTH, NUM_NEARBY_SLOTS, NEARBY_SLOT_WIDTH
    )
    neighbor = nearby_full.permute(0, 2, 1, 3).reshape(
        batch_size, NUM_NEARBY_SLOTS, HISTORY_LENGTH * NEARBY_SLOT_WIDTH
    )  # [B, 4, 84]

    lidar_full = _group(_LIDAR_OFFSET, _LIDAR_WIDTH)  # [B, 21, 240]
    lidar_full = lidar_full.reshape(batch_size, HISTORY_LENGTH, NUM_SECTORS, SECTOR_WIDTH)
    sector = lidar_full.permute(0, 2, 1, 3).reshape(
        batch_size, NUM_SECTORS, HISTORY_LENGTH * SECTOR_WIDTH
    )  # [B, 30, 168]

    return LidarTokenizerGroups(ego, navigation, side, lane, neighbor, sector), mask


def circular_positional_encoding(
    num_sectors: int, dim: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Fixed sin/cos encoding, exactly periodic with period `num_sectors`.

    `pe(i)` is `[sin(theta_i), cos(theta_i)]` tiled to `dim`, with
    `theta_i = 2*pi*i/num_sectors`. A single fundamental frequency keeps the
    pairwise Euclidean distance a strictly monotone function of the circular
    sector distance `min(|i-j|, num_sectors-|i-j|)` (`AC-006`; `REQ-007`).
    """

    if dim % 2 != 0:
        raise ValueError("Circular positional encoding dimension must be even")
    indices = torch.arange(num_sectors, device=device, dtype=torch.float32)
    theta = 2.0 * torch.pi * indices / num_sectors
    pair = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)  # [num_sectors, 2]
    return pair.repeat(1, dim // 2)  # [num_sectors, dim]
