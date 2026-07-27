from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thesis_rl.curriculum.config import ScenarioAclMabConfig

# DEC-006: per-arm reward-scale EMA that removes the cross-arm reward-magnitude
# confound identified in FIND-004 (docs/implementation/
# automatic_curriculum_learning_v1.2_exec_plan.md). The initial value is shared
# across all arms, not tuned per arm, so normalization is a no-op until each
# arm's estimate has accumulated enough episodes to diverge from this prior.
_INITIAL_REWARD_SCALE = 1.0
_MIN_REWARD_SCALE = 1e-3


@dataclass
class ScenarioArmBandit:
    """EMA teacher over the six frozen ScenarioNet semantic arms."""

    config: ScenarioAclMabConfig
    scores: np.ndarray = field(init=False, repr=False)
    target_scores: np.ndarray = field(init=False, repr=False)
    reward_scale: np.ndarray = field(init=False, repr=False)
    update_count: int = 0
    state_schema: str = field(init=False, default="acl_ema_v2")

    def __post_init__(self) -> None:
        if self.config.update_method != "ema":
            raise ValueError("ScenarioArmBandit only supports update_method='ema'.")
        if not 0.0 < float(self.config.alpha) <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1].")
        if not 0.0 <= float(self.config.initial_score) <= 1.0:
            raise ValueError("EMA initial_score must be in [0, 1].")
        if float(self.config.temperature) <= 0.0:
            raise ValueError("MAB temperature must be > 0.")
        self.scores = np.full(
            int(self.config.num_arms), float(self.config.initial_score), dtype=np.float64
        )
        self.target_scores = self.scores.copy()
        self.reward_scale = np.full(
            int(self.config.num_arms), _INITIAL_REWARD_SCALE, dtype=np.float64
        )

    @property
    def weights(self) -> np.ndarray:
        """Compatibility alias for diagnostics; scores are not logits."""
        return self.scores

    def probabilities(self, eligible_mask: np.ndarray | None = None) -> np.ndarray:
        """Return the Generate selection distribution.

        ``eligible_mask`` restricts sampling to arms that currently have a
        fresh catalog record (ACL-SN-EXH-001, `DEC-EXH-001`/`DEC-EXH-002`,
        ADR-028). A `False` entry gets exactly probability zero; the
        exploration floor `eta/K` is renormalized over the eligible subset so
        every *eligible* arm keeps a non-zero floor instead of every arm.
        Omitting the mask (or passing an all-`True` mask) reproduces the
        unrestricted ACL v1.1 REQ-002 distribution.
        """

        scores = np.asarray(
            self.target_scores if self.config.use_target_mab else self.scores,
            dtype=np.float64,
        )
        num_arms = int(self.config.num_arms)
        if scores.shape != (num_arms,) or not np.isfinite(scores).all():
            raise ValueError("MAB scores must be a finite vector with num_arms entries.")
        if eligible_mask is None:
            mask = np.ones(num_arms, dtype=bool)
        else:
            mask = np.asarray(eligible_mask, dtype=bool)
            if mask.shape != (num_arms,):
                raise ValueError("MAB eligible_mask must have num_arms entries.")
            if not mask.any():
                raise ValueError("At least one arm must be eligible for Generate sampling.")
        tau = float(self.config.temperature)
        logits = np.where(mask, scores / tau, -np.inf)
        logits -= float(np.max(logits))
        softmax = np.where(mask, np.exp(logits), 0.0)
        softmax /= float(np.sum(softmax))
        eta = float(self.config.eta)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("MAB eta must be in [0, 1].")
        num_eligible = int(mask.sum())
        probabilities = np.where(mask, (1.0 - eta) * softmax + eta / num_eligible, 0.0)
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("MAB probabilities are not finite or do not sum to one.")
        return probabilities

    def sample_arm(
        self, rng: np.random.Generator, eligible_mask: np.ndarray | None = None
    ) -> tuple[int, np.ndarray]:
        probabilities = self.probabilities(eligible_mask)
        return int(rng.choice(np.arange(len(probabilities)), p=probabilities)), probabilities

    def update(
        self,
        *,
        arm_index: int,
        normalized_usefulness: float,
        selection_probability: float | None = None,
    ) -> None:
        del selection_probability  # EMA deliberately has no importance correction.
        index = int(arm_index)
        value = float(normalized_usefulness)
        if not 0 <= index < len(self.scores):
            raise ValueError(f"MAB arm_index out of range: {index}")
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Normalized learning potential must be finite and in [0, 1].")
        alpha = float(self.config.alpha)
        self.scores[index] = (1.0 - alpha) * self.scores[index] + alpha * value
        self.scores[index] = float(np.clip(self.scores[index], 0.0, 1.0))
        self.update_count += 1
        if (
            self.config.use_target_mab
            and self.update_count % int(self.config.target_sync_interval) == 0
        ):
            self.target_scores = self.scores.copy()

    def reward_scale_estimate(self, arm_index: int) -> float:
        """Return the clamped per-arm reward-scale EMA (DEC-006)."""
        index = int(arm_index)
        if not 0 <= index < len(self.reward_scale):
            raise ValueError(f"MAB arm_index out of range: {index}")
        return float(max(float(self.reward_scale[index]), _MIN_REWARD_SCALE))

    def normalize_learning_potential_by_reward_scale(
        self, arm_index: int, raw_learning_potential: float
    ) -> float:
        """Divide a raw episode LP by the arm's reward-scale estimate (DEC-006).

        Uses the estimate as it stands *before* this episode's own
        contribution is folded in via ``update_reward_scale``, so the episode
        cannot normalize itself.
        """
        value = float(raw_learning_potential)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Raw learning potential must be finite and non-negative.")
        return value / self.reward_scale_estimate(arm_index)

    def update_reward_scale(self, arm_index: int, episode_reward: float) -> None:
        """Update the per-arm reward-scale EMA (DEC-006). Generate episodes only."""
        index = int(arm_index)
        if not 0 <= index < len(self.reward_scale):
            raise ValueError(f"MAB arm_index out of range: {index}")
        magnitude = abs(float(episode_reward))
        if not np.isfinite(magnitude):
            raise ValueError("Episode reward must be finite.")
        alpha = float(self.config.alpha)
        self.reward_scale[index] = (1.0 - alpha) * float(
            self.reward_scale[index]
        ) + alpha * magnitude

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.state_schema,
            "scores": [float(value) for value in self.scores.tolist()],
            "target_scores": [float(value) for value in self.target_scores.tolist()],
            "reward_scale": [float(value) for value in self.reward_scale.tolist()],
            "update_count": int(self.update_count),
        }

    @classmethod
    def from_state_dict(
        cls,
        config: ScenarioAclMabConfig,
        state: dict[str, object],
    ) -> "ScenarioArmBandit":
        if state.get("schema") != "acl_ema_v2":
            raise ValueError(
                "Incompatible Scenario ACL MAB checkpoint: expected schema 'acl_ema_v2'. "
                "DEC-006 added a per-arm reward-scale estimator; 'acl_ema_v1' checkpoints "
                "predate it and cannot be resumed."
            )
        bandit = cls(config)
        scores = state.get("scores")
        if not isinstance(scores, list) or len(scores) != int(config.num_arms):
            raise ValueError("Incompatible Scenario ACL MAB scores state.")
        bandit.scores = np.asarray(scores, dtype=np.float64)
        target_scores = state.get("target_scores", scores)
        if not isinstance(target_scores, list) or len(target_scores) != int(config.num_arms):
            raise ValueError("Incompatible Scenario ACL MAB target scores state.")
        bandit.target_scores = np.asarray(target_scores, dtype=np.float64)
        if not np.isfinite(bandit.scores).all() or not (
            (0.0 <= bandit.scores).all() and (bandit.scores <= 1.0).all()
        ):
            raise ValueError("Scenario ACL MAB checkpoint scores must be in [0, 1].")
        if not np.isfinite(bandit.target_scores).all() or not (
            (0.0 <= bandit.target_scores).all() and (bandit.target_scores <= 1.0).all()
        ):
            raise ValueError("Scenario ACL MAB checkpoint target scores must be in [0, 1].")
        reward_scale = state.get("reward_scale")
        if not isinstance(reward_scale, list) or len(reward_scale) != int(config.num_arms):
            raise ValueError("Incompatible Scenario ACL MAB reward-scale state.")
        bandit.reward_scale = np.asarray(reward_scale, dtype=np.float64)
        if not np.isfinite(bandit.reward_scale).all() or not (bandit.reward_scale >= 0.0).all():
            raise ValueError(
                "Scenario ACL MAB checkpoint reward-scale values must be finite and non-negative."
            )
        bandit.update_count = int(state.get("update_count", 0))
        return bandit
