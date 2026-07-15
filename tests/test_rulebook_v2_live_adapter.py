from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.context.live_adapter import LiveSnapshotAdapter, LiveSnapshotSources
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot
import pytest


def _actor(actor_id: str) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id, ActorClass.VEHICLE, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0),
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1))), "lane", 20.0,
    )


def test_live_snapshot_adapter_uses_only_explicit_sources():
    env = {"scenario": "s", "step": 3, "time": 0.3}
    sources = LiveSnapshotSources(
        scenario_id=lambda value: value["scenario"],
        step_index=lambda value: value["step"],
        sim_time_s=lambda value: value["time"],
        ego=lambda _value: _actor("ego"),
        actors=lambda _value: (_actor("other"),),
        contact_onset_records=lambda _value: (),
        active_contact_ids=lambda _value: frozenset(),
        signal_states_by_physical_id=lambda _value: {"signal": "GREEN"},
    )
    snapshot = LiveSnapshotAdapter(sources).capture(env)
    assert snapshot.scenario_id == "s"
    assert snapshot.step_index == 3
    assert snapshot.ego.actor_id == "ego"
    assert snapshot.signal_states_by_physical_id["signal"] == "GREEN"


def test_live_snapshot_sources_reject_partial_or_unknown_provider_mappings():
    with pytest.raises(ValueError, match="Missing"):
        LiveSnapshotSources.from_mapping({})
    providers = {name: lambda _env: None for name in LiveSnapshotSources.__dataclass_fields__}
    providers["unexpected"] = lambda _env: None
    with pytest.raises(ValueError, match="Unknown"):
        LiveSnapshotSources.from_mapping(providers)
    with pytest.raises(TypeError, match="callable"):
        LiveSnapshotSources(*(None,) * 8)
