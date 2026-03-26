from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class NeuralAdapter:
    """Skeleton adapter for learned action transformation.

    This is a placeholder for future neural adapters that map planner outputs
    (e.g., waypoints or high-level intents) to low-level MetaDrive actions.
    """

    is_neural = True
    requires_training = True

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        clip: bool = True,
        expected_shape: tuple[int, ...] = (2,),
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        update_interval: int = 1,
        buffer_capacity: int = 10000,
        device: str = "cpu",
    ) -> None:
        self.low = low
        self.high = high
        self.clip = clip
        self.expected_shape = expected_shape
        self.input_dim = int(np.prod(expected_shape))

        requested = device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Tanh(),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.update_interval = max(1, update_interval)
        self.buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=buffer_capacity)
        self.step_count = 0
        self.last_loss = float("nan")

    def __call__(self, planner_output: Any) -> np.ndarray:
        planner_arr = np.asarray(planner_output, dtype=np.float32)
        if planner_arr.shape != self.expected_shape:
            raise ValueError(f"Expected action shape {self.expected_shape}, got {planner_arr.shape}")

        x = planner_arr.reshape(1, -1)
        x_tensor = torch.from_numpy(x).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy().reshape(self.expected_shape)

        action = pred.astype(np.float32)
        if action.shape != self.expected_shape:
            raise ValueError(f"Expected action shape {self.expected_shape}, got {action.shape}")
        if self.clip:
            action = np.clip(action, self.low, self.high)

        # Identity target for now: learn a smooth adapter around direct-action.
        target = np.clip(planner_arr, self.low, self.high) if self.clip else planner_arr
        self.buffer.append((planner_arr.copy(), target.copy()))
        self.step_count += 1
        return action

    def begin_training(self) -> None:
        self.model.train()
        self.step_count = 0
        return None

    def maybe_update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return None
        if self.step_count % self.update_interval != 0:
            return None

        idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        x_np = np.stack([item[0].reshape(-1) for item in batch], axis=0).astype(np.float32)
        y_np = np.stack([item[1].reshape(-1) for item in batch], axis=0).astype(np.float32)

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu().item())
        return None

    def end_training(self) -> None:
        self.model.eval()
        return None

    def save(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "expected_shape": self.expected_shape,
                "step_count": self.step_count,
            },
            str(checkpoint),
        )
        return None

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = int(checkpoint.get("step_count", 0))
        self.model.eval()
        return None
