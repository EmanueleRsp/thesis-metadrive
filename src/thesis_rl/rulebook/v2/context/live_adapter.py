"""Strict live-environment adapter hooks for Rulebook v2 snapshots.

Source-specific MetaDrive/ScenarioNet code supplies the callables in
``LiveSnapshotSources``.  The v2 core never guesses attribute names and never
falls back to observations or future trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from thesis_rl.rulebook.v2.context.snapshotter import capture_env_snapshot
from thesis_rl.rulebook.v2.types import ActorSnapshot, ContactOnsetRecord, EnvSnapshot


@dataclass(frozen=True, slots=True)
class LiveSnapshotSources:
    """Environment-owned providers for one canonical control-step snapshot."""

    scenario_id: Callable[[Any], str]
    step_index: Callable[[Any], int]
    sim_time_s: Callable[[Any], float]
    ego: Callable[[Any], ActorSnapshot]
    actors: Callable[[Any], tuple[ActorSnapshot, ...]]
    contact_onset_records: Callable[[Any], tuple[ContactOnsetRecord, ...]]
    active_contact_ids: Callable[[Any], frozenset[str]]
    signal_states_by_physical_id: Callable[[Any], Mapping[str, str]]

    def __post_init__(self) -> None:
        providers = (
            self.scenario_id, self.step_index, self.sim_time_s, self.ego,
            self.actors, self.contact_onset_records, self.active_contact_ids,
            self.signal_states_by_physical_id,
        )
        if any(not callable(provider) for provider in providers):
            raise TypeError("Every LiveSnapshotSources field must be callable")

    @classmethod
    def from_mapping(cls, providers: Mapping[str, Callable[[Any], object]]) -> "LiveSnapshotSources":
        """Construct sources from an adapter mapping, rejecting missing keys."""
        required = tuple(cls.__dataclass_fields__)
        missing = tuple(name for name in required if name not in providers)
        unknown = tuple(sorted(set(providers).difference(required)))
        if missing:
            raise ValueError(f"Missing live snapshot providers: {missing}")
        if unknown:
            raise ValueError(f"Unknown live snapshot providers: {unknown}")
        return cls(**{name: providers[name] for name in required})


class LiveSnapshotAdapter:
    """Build immutable snapshots using only explicitly supplied live hooks."""

    def __init__(self, sources: LiveSnapshotSources) -> None:
        self.sources = sources

    def capture(self, env: Any) -> EnvSnapshot:
        sources = self.sources
        return capture_env_snapshot(
            scenario_id=sources.scenario_id(env),
            step_index=sources.step_index(env),
            sim_time_s=sources.sim_time_s(env),
            ego=sources.ego(env),
            actors=sources.actors(env),
            contact_onset_records=sources.contact_onset_records(env),
            active_contact_ids=sources.active_contact_ids(env),
            signal_states_by_physical_id=sources.signal_states_by_physical_id(env),
        )
