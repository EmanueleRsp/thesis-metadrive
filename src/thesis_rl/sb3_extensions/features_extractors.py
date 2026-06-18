"""SB3 feature extractors backed by thesis-specific encoder modules."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from thesis_rl.agent.planners.core.utils import build_encoder_for_env, to_plain_dict


class ThesisEncoderFeatureExtractor(BaseFeaturesExtractor):
    """Expose thesis encoder modules through the SB3 features-extractor API."""

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        cfg_encoder: Any,
        cfg_obs: Any | None = None,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(
                "ThesisEncoderFeatureExtractor currently requires a Box observation "
                f"space, got {type(observation_space).__name__}."
            )

        obs_dim = int(torch.as_tensor(observation_space.shape).prod().item())
        encoder_cfg = to_plain_dict(cfg_encoder)
        obs_cfg = {} if cfg_obs is None else to_plain_dict(cfg_obs)
        encoder = build_encoder_for_env(encoder_cfg, obs_cfg, obs_dim=obs_dim)

        super().__init__(observation_space, features_dim=int(encoder.output_dim))
        self.cfg_encoder = encoder_cfg
        self.cfg_obs = obs_cfg
        self.encoder = encoder

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat_obs = observations.float()
        if flat_obs.ndim > 2:
            flat_obs = torch.flatten(flat_obs, start_dim=1)
        return self.encoder(flat_obs)
