"""Immutable live snapshot capture at Rulebook control-step boundaries."""

from __future__ import annotations

from math import isfinite
from typing import Mapping

from thesis_rl.rulebook.v2.types import ActorSnapshot, ContactOnsetRecord, EnvSnapshot


def capture_env_snapshot(
    *,
    scenario_id: str,
    step_index: int,
    sim_time_s: float,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    contact_onset_records: tuple[ContactOnsetRecord, ...],
    active_contact_ids: frozenset[str],
    signal_states_by_physical_id: Mapping[str, str],
) -> EnvSnapshot:
    """Build a complete immutable snapshot and reject duplicate actor identity."""

    if not scenario_id or step_index < 0 or not isfinite(sim_time_s):
        raise ValueError("Snapshot identity and simulation time must be valid")
    actor_ids = [actor.actor_id for actor in (ego, *actors)]
    if any(not actor_id for actor_id in actor_ids):
        raise ValueError("Snapshot actor IDs must be non-empty")
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("Snapshot contains duplicate actor IDs")
    if any(not record.actor_id for record in contact_onset_records):
        raise ValueError("Contact onset actor IDs must be non-empty")
    return EnvSnapshot(
        scenario_id=scenario_id,
        step_index=step_index,
        sim_time_s=sim_time_s,
        ego=ego,
        actors=tuple(actors),
        contact_onset_records=tuple(contact_onset_records),
        active_contact_ids=frozenset(active_contact_ids),
        signal_states_by_physical_id=dict(signal_states_by_physical_id),
    )
