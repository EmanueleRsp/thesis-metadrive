from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thesis_rl.curriculum.config import ScenarioAclMabConfig


@dataclass
class GeneratorArmBandit:
    config: ScenarioAclMabConfig
    weights: np.ndarray = field(init=False, repr=False)
    target_weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        initial = float(self.config.initial_weight)
        decay = float(self.config.initial_weight_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("initial_weight_decay must be in (0, 1].")
        # ``weights`` are softmax logits.  Log-spaced offsets therefore give
        # an exact exponential probability schedule: p(A_i)/p(A_0)=decay**i.
        offsets = np.arange(int(self.config.num_arms), dtype=np.float64) * np.log(decay)
        self.weights = initial + offsets
        self.target_weights = self.weights.copy()

    def probabilities(self) -> np.ndarray:
        logits = np.clip(
            self.weights,
            float(self.config.weight_clip_min),
            float(self.config.weight_clip_max),
        )
        logits = logits - float(np.max(logits))
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits)
        eta = float(self.config.eta)
        k = int(self.config.num_arms)
        return ((1.0 - eta) * softmax) + (eta / float(k))

    def sample_arm(
        self,
        rng: np.random.Generator,
    ) -> tuple[int, np.ndarray]:
        probs = self.probabilities()
        arm_index = int(rng.choice(np.arange(len(probs)), p=probs))
        return arm_index, probs

    def update(self, *, arm_index: int, normalized_usefulness: float, selection_probability: float) -> None:
        p = max(float(selection_probability), 1e-8)
        feedback = float(normalized_usefulness) / p
        updated = float(self.target_weights[int(arm_index)]) + (
            float(self.config.alpha) * feedback
        )
        updated = min(
            max(updated, float(self.config.weight_clip_min)),
            float(self.config.weight_clip_max),
        )
        self.target_weights[int(arm_index)] = updated
        self.update_count += 1

        if not bool(self.config.use_target_mab):
            self.weights = self.target_weights.copy()
            return

        if self.update_count % int(self.config.target_sync_interval) == 0:
            self.weights = self.target_weights.copy()

    def state_dict(self) -> dict[str, object]:
        return {
            "weights": [float(x) for x in self.weights.tolist()],
            "target_weights": [float(x) for x in self.target_weights.tolist()],
            "update_count": int(self.update_count),
        }

    @classmethod
    def from_state_dict(
        cls,
        config: ScenarioAclMabConfig,
        state: dict[str, object],
    ) -> "GeneratorArmBandit":
        bandit = cls(config)
        weights = state.get("weights")
        target_weights = state.get("target_weights")
        if isinstance(weights, list) and len(weights) == int(config.num_arms):
            bandit.weights = np.asarray(weights, dtype=np.float64)
        if isinstance(target_weights, list) and len(target_weights) == int(config.num_arms):
            bandit.target_weights = np.asarray(target_weights, dtype=np.float64)
        bandit.update_count = int(state.get("update_count", 0))
        return bandit
