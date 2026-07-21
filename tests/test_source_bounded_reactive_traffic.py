from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from thesis_rl.envs.scenario_traffic_manager import (
    SourceBoundScenarioTrafficManager,
    source_track_is_valid_at_step,
)


def test_source_track_validity_accepts_only_current_valid_samples() -> None:
    track = {"state": {"valid": np.asarray([True, False], dtype=bool)}}

    assert source_track_is_valid_at_step(track, 0) is True
    assert source_track_is_valid_at_step(track, 1) is False


def test_source_track_validity_rejects_out_of_range_and_malformed_tracks() -> None:
    track = {"state": {"valid": np.asarray([True], dtype=bool)}}

    assert source_track_is_valid_at_step(track, -1) is False
    assert source_track_is_valid_at_step(track, 1) is False
    assert source_track_is_valid_at_step(None, 0) is False
    assert source_track_is_valid_at_step({}, 0) is False


def test_reactive_manager_skips_actuation_and_queues_expired_source(
    monkeypatch,
) -> None:
    class FakeVehicle:
        def __init__(self, identifier: str) -> None:
            self.id = identifier
            self.name = identifier
            self.actions: list[object] = []

        def before_step(self, action: object) -> None:
            self.actions.append(action)

    class FakePolicy:
        arrive_destination = False
        policy_index = 0

        def __init__(self) -> None:
            self.calls: list[bool] = []

        def act(self, do_speed_control: bool) -> str:
            self.calls.append(do_speed_control)
            return "action"

    valid_vehicle = FakeVehicle("valid")
    expired_vehicle = FakeVehicle("expired")
    valid_policy = FakePolicy()
    expired_policy = FakePolicy()
    engine = SimpleNamespace(
        episode_step=1,
        data_manager=SimpleNamespace(
            current_scenario={
                "tracks": {
                    "valid": {"state": {"valid": np.asarray([True, True])}},
                    "expired": {"state": {"valid": np.asarray([True])}},
                }
            }
        ),
    )
    engine.has_policy = lambda object_id, _: object_id in {"valid", "expired"}
    engine.get_policy = lambda object_id: {
        "valid": valid_policy,
        "expired": expired_policy,
    }[object_id]
    monkeypatch.setattr("metadrive.engine.engine_utils.get_engine", lambda: engine)

    manager = object.__new__(SourceBoundScenarioTrafficManager)
    manager.spawned_objects = {"valid": valid_vehicle, "expired": expired_vehicle}
    manager._obj_id_to_scenario_id = {"valid": "valid", "expired": "expired"}

    manager.before_step()

    assert valid_policy.calls == [False]
    assert valid_vehicle.actions == ["action"]
    assert expired_policy.calls == []
    assert expired_vehicle.actions == []
    assert manager._obj_to_clean_this_frame == ["expired"]
