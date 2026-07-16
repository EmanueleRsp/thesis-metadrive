"""Observation-safe view of canonical Rulebook v2 state.

The type intentionally excludes RulebookResult, costs, margins, statuses and
rewards. It is immutable so an observation builder can consume only one
committed control-step snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

from thesis_rl.rulebook.v2.types import EpisodeCache, EnvSnapshot, RulebookMemory


@dataclass(frozen=True, slots=True)
class CausalSceneContext:
    """Environment-owned, committed causal input for one observation step."""

    episode_cache: EpisodeCache
    snapshot: EnvSnapshot
    memory: RulebookMemory

    def __post_init__(self) -> None:
        if self.episode_cache.scenario_id != self.snapshot.scenario_id:
            raise ValueError(
                "CausalSceneContext requires cache and snapshot from the same scenario."
            )

    @property
    def task_route(self):
        return self.episode_cache.task_route

    @property
    def conflict_zones(self):
        return self.episode_cache.conflict_zones

    @property
    def traffic_controls(self):
        return self.episode_cache.traffic_control_catalog
