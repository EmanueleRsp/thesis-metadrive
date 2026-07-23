"""Rollout buffers for the local SB3 bridge."""

try:  # Keep configuration validation importable without optional SB3 runtime deps.
    from thesis_rl.sb3_extensions.rollout.masked import MaskedRolloutBuffer
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the project env
    MaskedRolloutBuffer = None  # type: ignore[assignment,misc]

__all__ = ["MaskedRolloutBuffer"]
