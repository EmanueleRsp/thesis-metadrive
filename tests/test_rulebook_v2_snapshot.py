from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2 import ActorClass, ContactOnsetBuffer
from thesis_rl.rulebook.v2.context.snapshotter import capture_env_snapshot
from thesis_rl.rulebook.v2.types import ActorSnapshot, ContactOnsetRecord


def _actor(actor_id: str) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id=actor_id,
        actor_class=ActorClass.VEHICLE,
        position_xy=(0.0, 0.0),
        position_z=0.0,
        heading_rad=0.0,
        velocity_xy=(0.0, 0.0),
        footprint=Polygon(((0, 0), (1, 0), (1, 1), (0, 0))),
        live_lane_id="lane",
        configured_speed_cap_mps=20.0,
    )


def test_snapshot_capture_is_complete_and_rejects_duplicate_actor_ids() -> None:
    snapshot = capture_env_snapshot(
        scenario_id="scenario",
        step_index=1,
        sim_time_s=0.1,
        ego=_actor("ego"),
        actors=(_actor("other"),),
        contact_onset_records=(),
        active_contact_ids=frozenset(),
        signal_states_by_physical_id={"signal": "GREEN"},
    )
    assert snapshot.ego.actor_id == "ego"
    assert snapshot.signal_states_by_physical_id["signal"] == "GREEN"
    with pytest.raises(ValueError, match="duplicate"):
        capture_env_snapshot(
            scenario_id="scenario",
            step_index=1,
            sim_time_s=0.1,
            ego=_actor("ego"),
            actors=(_actor("ego"),),
            contact_onset_records=(),
            active_contact_ids=frozenset(),
            signal_states_by_physical_id={},
        )


def test_contact_buffer_preserves_points_and_drains_atomically() -> None:
    buffer = ContactOnsetBuffer()
    buffer.clear_control_step()
    buffer.record_onset(ContactOnsetRecord("other", ActorClass.VEHICLE, (1.0, 0.0), (1.0, 0.0)))
    buffer.record_onset(ContactOnsetRecord("other", ActorClass.VEHICLE, (1.1, 0.0), (1.0, 0.0)))
    drained = buffer.drain()
    assert len(drained) == 2
    assert buffer.drain() == ()
