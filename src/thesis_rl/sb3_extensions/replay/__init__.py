"""Transition replay configuration and buffers for the local SB3 bridge."""

from thesis_rl.sb3_extensions.replay.config import (
    TransitionReplayConfig,
    resolve_transition_replay_config,
)
try:  # Keep configuration validation importable without optional SB3 runtime deps.
    from thesis_rl.sb3_extensions.replay.prioritized import PrioritizedNStepReplayBuffer
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the project env
    PrioritizedNStepReplayBuffer = None  # type: ignore[assignment,misc]

__all__ = [
    "PrioritizedNStepReplayBuffer",
    "TransitionReplayConfig",
    "resolve_transition_replay_config",
]
