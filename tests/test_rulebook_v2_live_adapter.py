from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.context.live_adapter import (
    LiveSnapshotAdapter,
    LiveSnapshotSources,
    actor_snapshot_from_payload,
    contact_onset_from_payload,
    install_collision_callback_hook,
    wrap_collision_callback,
)
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


def test_actor_snapshot_payload_builds_canonical_obb_and_rejects_incomplete_data():
    actor = actor_snapshot_from_payload(
        {
            "actor_id": "vehicle-7",
            "actor_class": "vehicle",
            "position_xy": (1.0, 2.0),
            "position_z": 0.5,
            "heading_rad": 1.5707963267948966,
            "velocity_xy": (0.0, 4.0),
            "length_m": 4.5,
            "width_m": 2.0,
            "live_lane_id": "lane-3",
            "configured_speed_cap_mps": 13.0,
        }
    )
    assert actor.actor_id == "vehicle-7"
    assert actor.actor_class is ActorClass.VEHICLE
    assert actor.live_lane_id == "lane-3"
    assert actor.footprint.area == pytest.approx(9.0)
    with pytest.raises(ValueError, match="missing fields"):
        actor_snapshot_from_payload({"actor_id": "vehicle-7"})
    with pytest.raises(ValueError, match="finite"):
        actor_snapshot_from_payload(
            {
                "actor_id": "vehicle-7",
                "actor_class": "vehicle",
                "position_xy": (float("nan"), 0.0),
                "position_z": 0.0,
                "heading_rad": 0.0,
                "velocity_xy": (0.0, 0.0),
                "length_m": 4.0,
                "width_m": 2.0,
            }
        )


def test_contact_payload_requires_unit_ego_to_other_normal():
    record = contact_onset_from_payload(
        {
            "actor_id": "ped-1",
            "actor_class": ActorClass.PEDESTRIAN,
            "contact_point_xy": (2.0, 3.0),
            "normal_ego_to_other_xy": (1.0, 0.0),
        }
    )
    assert record.actor_class is ActorClass.PEDESTRIAN
    assert record.normal_ego_to_other_xy == (1.0, 0.0)
    with pytest.raises(ValueError, match="unit"):
        contact_onset_from_payload(
            {
                "actor_id": "ped-1",
                "actor_class": "pedestrian",
                "contact_point_xy": (2.0, 3.0),
                "normal_ego_to_other_xy": (2.0, 0.0),
            }
        )
    inverted = contact_onset_from_payload(
        {
            "actor_id": "ped-1",
            "actor_class": "pedestrian",
            "contact_point_xy": (2.0, 3.0),
            "normal_xy": (-1.0, 0.0),
            "normal_orientation": "other_to_ego",
        }
    )
    assert inverted.normal_ego_to_other_xy == (1.0, 0.0)


def test_collision_wrapper_preserves_vendor_return_and_order():
    calls: list[str] = []

    def vendor(value: int) -> int:
        calls.append(f"vendor:{value}")
        return value + 1

    wrapped = wrap_collision_callback(vendor, lambda value: calls.append(f"hook:{value}"))
    assert wrapped(4) == 5
    assert calls == ["vendor:4", "hook:4"]


def test_collision_hook_installs_wrapped_callback_at_existing_world_boundary():
    class FakeWorld:
        def setContactAddedCallback(self, callback):
            self.callback = callback

    world = FakeWorld()
    observed: list[int] = []
    install_collision_callback_hook(
        world,
        original_callback=lambda value: value + 2,
        observer=lambda value: observed.append(value),
        callback_object_factory=lambda callback: callback,
    )
    assert world.callback(5) == 7
    assert observed == [5]
