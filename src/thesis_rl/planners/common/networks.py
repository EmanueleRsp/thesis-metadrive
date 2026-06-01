from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from thesis_rl.planners.common.utils import safe_atanh


class DeterministicActor(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_head = nn.Linear(int(decoder.output_dim), int(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = self.decoder(z)
        return torch.tanh(self.action_head(h))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        action_dim: int,
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
        state_dependent_std: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        hidden_dim = int(decoder.output_dim)
        self.mu_head = nn.Linear(hidden_dim, int(action_dim))
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(hidden_dim, int(action_dim))
            self.log_std_param = None
        else:
            self.log_std_head = None
            self.log_std_param = nn.Parameter(torch.zeros(int(action_dim)))
        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])

    def _dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        h = self.decoder(z)
        mu = self.mu_head(h)
        if self.log_std_head is not None:
            log_std = self.log_std_head(h)
        else:
            assert self.log_std_param is not None
            log_std = self.log_std_param.unsqueeze(0).expand_as(mu)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._dist_params(obs)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self._dist_params(obs)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        u = safe_atanh(actions)
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - actions.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class TwinQCritic(nn.Module):
    def __init__(self, encoder: nn.Module, decoder_q1: nn.Module, decoder_q2: nn.Module, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder_q1 = decoder_q1
        self.decoder_q2 = decoder_q2
        self.q1_head = nn.Linear(int(decoder_q1.output_dim), 1)
        self.q2_head = nn.Linear(int(decoder_q2.output_dim), 1)
        self.action_dim = int(action_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        x = torch.cat([z, actions], dim=-1)
        q1 = self.q1_head(self.decoder_q1(x))
        q2 = self.q2_head(self.decoder_q2(x))
        return q1, q2

    def q1(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        x = torch.cat([z, actions], dim=-1)
        return self.q1_head(self.decoder_q1(x))


class ValueNet(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.value_head = nn.Linear(int(decoder.output_dim), 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = self.decoder(z)
        return self.value_head(h)
