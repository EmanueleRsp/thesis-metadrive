from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.reset()

    def reset(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.actions[i] = np.asarray(action, dtype=np.float32)
        self.rewards[i, 0] = float(reward)
        self.dones[i, 0] = float(done)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        n = int(obs.shape[0])
        for i in range(n):
            self.add(
                obs=obs[i],
                action=actions[i],
                reward=float(rewards[i]),
                done=bool(dones[i]),
                next_obs=next_obs[i],
            )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from empty replay buffer.")
        batch = min(int(batch_size), self.size)
        idx = np.random.randint(0, self.size, size=batch)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "next_obs": self.next_obs,
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        self.capacity = int(payload["capacity"])
        self.obs_dim = int(payload["obs_dim"])
        self.action_dim = int(payload["action_dim"])
        self.obs = np.asarray(payload["obs"], dtype=np.float32)
        self.actions = np.asarray(payload["actions"], dtype=np.float32)
        self.rewards = np.asarray(payload["rewards"], dtype=np.float32)
        self.dones = np.asarray(payload["dones"], dtype=np.float32)
        self.next_obs = np.asarray(payload["next_obs"], dtype=np.float32)
        self.ptr = int(payload["ptr"])
        self.size = int(payload["size"])


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.obs = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding new rollout.")
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(actions, dtype=np.float32)
        self.rewards[self.pos] = np.asarray(rewards, dtype=np.float32)
        self.dones[self.pos] = np.asarray(dones, dtype=np.float32)
        self.values[self.pos] = np.asarray(values, dtype=np.float32)
        self.log_probs[self.pos] = np.asarray(log_probs, dtype=np.float32)
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        last_values_np = np.asarray(last_values, dtype=np.float32)
        last_dones_np = np.asarray(last_dones, dtype=np.float32)
        gae = np.zeros((self.n_envs,), dtype=np.float32)

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones_np
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    def iter_minibatches(self, batch_size: int, device: torch.device):
        if not self.full:
            raise RuntimeError("RolloutBuffer not full. Cannot iterate minibatches yet.")

        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        obs = self.obs.reshape(total, self.obs_dim)
        actions = self.actions.reshape(total, self.action_dim)
        log_probs = self.log_probs.reshape(total)
        returns = self.returns.reshape(total)
        advantages = self.advantages.reshape(total)
        values = self.values.reshape(total)

        for start in range(0, total, int(batch_size)):
            idx = indices[start : start + int(batch_size)]
            yield RolloutBatch(
                obs=torch.as_tensor(obs[idx], dtype=torch.float32, device=device),
                actions=torch.as_tensor(actions[idx], dtype=torch.float32, device=device),
                old_log_probs=torch.as_tensor(log_probs[idx], dtype=torch.float32, device=device),
                returns=torch.as_tensor(returns[idx], dtype=torch.float32, device=device),
                advantages=torch.as_tensor(advantages[idx], dtype=torch.float32, device=device),
                old_values=torch.as_tensor(values[idx], dtype=torch.float32, device=device),
            )
