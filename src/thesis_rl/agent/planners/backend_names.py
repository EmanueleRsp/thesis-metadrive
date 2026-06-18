from __future__ import annotations

import logging


_LOGGER = logging.getLogger(__name__)

LEGACY_PLANNER_BACKENDS = frozenset({"td3", "sac", "ppo"})
SB3_FORK_PLANNER_BACKENDS = frozenset({"td3_sb3", "sac_sb3", "ppo_sb3"})
_LEGACY_TO_SB3_FORK = {
    "td3": "td3_sb3",
    "sac": "sac_sb3",
    "ppo": "ppo_sb3",
}
_warned_legacy_backends: set[str] = set()


def planner_backend_family(planner_name: str) -> str:
    """Return the implementation family for a configured planner backend name."""

    name = str(planner_name).lower()
    if name in LEGACY_PLANNER_BACKENDS:
        return "legacy"
    if name in SB3_FORK_PLANNER_BACKENDS:
        return "sb3_fork"
    raise ValueError(f"Unsupported planner backend: {planner_name}")


def preferred_fork_backed_backend(planner_name: str) -> str:
    """Return the preferred fork-backed backend for the same algorithm family."""

    name = str(planner_name).lower()
    if name in SB3_FORK_PLANNER_BACKENDS:
        return name
    if name in _LEGACY_TO_SB3_FORK:
        return _LEGACY_TO_SB3_FORK[name]
    raise ValueError(f"Unsupported planner backend: {planner_name}")


def warn_if_legacy_backend(planner_name: str) -> None:
    """Emit a one-time warning when a temporary legacy backend is selected."""

    name = str(planner_name).lower()
    if planner_backend_family(name) != "legacy":
        return
    if name in _warned_legacy_backends:
        return

    preferred = preferred_fork_backed_backend(name)
    _LOGGER.warning(
        "Planner backend '%s' is a temporary legacy implementation during the "
        "SB3-fork migration. Prefer the fork-backed backend '%s' or canonical "
        "presets under `presets/agent/*_sb3` for new runs.",
        name,
        preferred,
    )
    _warned_legacy_backends.add(name)


def _reset_legacy_backend_warning_cache() -> None:
    """Testing helper for deterministic warning assertions."""

    _warned_legacy_backends.clear()
