from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.context.metadrive_live import (
    MetaDriveContactRecorder,
    actor_snapshot_from_metadrive,
    contact_onset_from_metadrive_contact,
    live_actor_objects,
    live_actor_snapshots,
    live_ego_snapshot,
    live_signal_states_by_physical_id,
    live_vehicle_objects,
)
from thesis_rl.rulebook.v2.types import ActorClass


class _Lane:
    index = ("road", "lane", 0)


class _Vehicle:
    id = "runtime-object"
    position = (1.0, 2.0)
    velocity = (3.0, 0.5)
    heading_theta = 0.25
    LENGTH = 4.5
    WIDTH = 2.0
    max_speed_m_s = 20.0
    lane = _Lane()

    def get_z(self):
        return 0.75


class _Traffic:
    obj_id_to_scenario_id = {"runtime-object": "scenario-object"}
    vehicles = {"runtime-object": _Vehicle()}
    ego_vehicle = _Vehicle()


class _Engine:
    traffic_manager = _Traffic()

    def get_objects(self):
        return {}


class _Env:
    engine = _Engine()


class _Light:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return {"object_state": self.state}


class _LightManager:
    _scenario_id_to_obj_id = {"signal-red": "runtime-red", "signal-go": "runtime-go"}
    spawned_objects = {
        "runtime-red": _Light("LANE_STATE_STOP"),
        "runtime-go": _Light("TRAFFIC_LIGHT_GREEN"),
    }


class _SignalEngine:
    light_manager = _LightManager()


class _SignalEnv:
    engine = _SignalEngine()


class _OtherVehicle(_Vehicle):
    id = "other-runtime"


class _Node:
    def __init__(self, obj):
        self.obj = obj


class _Manifold:
    def getPositionWorldOnA(self):
        return (0.0, 0.0, 0.0)

    def getPositionWorldOnB(self):
        return (2.0, 0.0, 0.0)


class _Contact:
    def __init__(self):
        self.node0 = _Node(_Vehicle())
        self.node1 = _Node(_OtherVehicle())
        self.manifold_point = _Manifold()

    def getNode0(self):
        return self.node0

    def getNode1(self):
        return self.node1

    def getManifoldPoint(self):
        return self.manifold_point


def _object_from_node(node):
    return node.obj


def test_metadrive_actor_normalization_uses_stable_scenario_identity():
    snapshot = actor_snapshot_from_metadrive(_Env(), _Vehicle())
    assert snapshot.actor_id == "scenario-object"
    assert snapshot.actor_class is ActorClass.VEHICLE
    assert snapshot.position_z == pytest.approx(0.75)
    assert snapshot.live_lane_id == "('road', 'lane', 0)"
    assert snapshot.footprint.area == pytest.approx(9.0)


def test_metadrive_vehicle_collection_is_deterministic_and_deduplicated():
    assert live_vehicle_objects(_Env()) == (_Traffic.vehicles["runtime-object"],)


def test_metadrive_ego_and_actor_snapshots_have_explicit_partition():
    assert live_ego_snapshot(_Env()).actor_id == "scenario-object"
    assert live_actor_snapshots(_Env()) == ()


def test_metadrive_actor_objects_include_public_registry_vru():
    class Cyclist(_Vehicle):
        id = "cyclist-runtime"

    class RegistryEngine(_Engine):
        def get_objects(self):
            return {"cyclist-runtime": Cyclist()}

    class RegistryEnv:
        engine = RegistryEngine()

    objects = live_actor_objects(RegistryEnv())
    assert {obj.id for obj in objects} == {"runtime-object", "cyclist-runtime"}
    snapshots = live_actor_snapshots(RegistryEnv())
    assert {snapshot.actor_class for snapshot in snapshots} == {ActorClass.CYCLIST}


def test_metadrive_actor_objects_fail_closed_on_unknown_registry_object():
    class UnknownObject:
        id = "unknown-runtime"

    class RegistryEngine(_Engine):
        def get_objects(self):
            return {"unknown-runtime": UnknownObject()}

    class RegistryEnv:
        engine = RegistryEngine()

    with pytest.raises(ValueError, match="Unsupported MetaDrive live object"):
        live_actor_objects(RegistryEnv())


def test_metadrive_actor_snapshot_partition_rejects_missing_ego():
    class NoEgoTraffic:
        vehicles = _Traffic.vehicles
        ego_vehicle = None

    class NoEgoEngine:
        traffic_manager = NoEgoTraffic()

    class NoEgoEnv:
        engine = NoEgoEngine()

    with pytest.raises(ValueError, match="configured ego"):
        live_actor_snapshots(NoEgoEnv())


def test_metadrive_signal_provider_reads_current_states_by_source_id():
    assert live_signal_states_by_physical_id(_SignalEnv()) == {
        "signal-go": "GREEN",
        "signal-red": "RED",
    }


def test_metadrive_signal_provider_keeps_unknown_state_explicit():
    class UnknownLightManager(_LightManager):
        spawned_objects = {
            "runtime-red": _Light("LANE_STATE_MYSTERY"),
            "runtime-go": _Light("LANE_STATE_GO"),
        }

    class UnknownSignalEngine:
        light_manager = UnknownLightManager()

    class UnknownSignalEnv:
        engine = UnknownSignalEngine()

    assert live_signal_states_by_physical_id(UnknownSignalEnv())["signal-red"] == "UNKNOWN"


def test_metadrive_actor_requires_vehicle_speed_cap():
    class NoCapVehicle(_Vehicle):
        max_speed_m_s = None

    with pytest.raises(ValueError, match="speed cap"):
        actor_snapshot_from_metadrive(_Env(), NoCapVehicle())


def test_metadrive_contact_normal_is_oriented_from_ego_to_other():
    env = _Env()
    _Traffic.obj_id_to_scenario_id["other-runtime"] = "other-scenario"
    record = contact_onset_from_metadrive_contact(
        _Contact(), env, object_from_node=_object_from_node
    )
    assert record.actor_id == "other-scenario"
    assert record.actor_class is ActorClass.VEHICLE
    assert record.contact_point_xy == (1.0, 0.0)
    assert record.normal_ego_to_other_xy == (1.0, 0.0)


def test_metadrive_contact_recorder_clears_per_control_step():
    recorder = MetaDriveContactRecorder(_Env(), object_from_node=_object_from_node)
    recorder.observe(_Contact())
    records, active = recorder.snapshot_contact_state()
    assert len(records) == 1
    assert active == frozenset({"other-scenario"})
    recorder.clear_control_step()
    assert recorder.snapshot_contact_state() == ((), frozenset())


def test_metadrive_contact_recorder_keeps_persistent_manifold_active():
    _Traffic.obj_id_to_scenario_id["other-runtime"] = "other-scenario"

    class World:
        manifolds = [_Contact()]

        def get_manifolds(self):
            return tuple(self.manifolds)

    class Physics:
        dynamic_world = World()

    class PersistentEngine:
        traffic_manager = _Traffic()
        physics_world = Physics()

    class PersistentEnv:
        engine = PersistentEngine()

    recorder = MetaDriveContactRecorder(PersistentEnv(), object_from_node=_object_from_node)
    recorder.observe(_Contact())
    first_records, first_active = recorder.snapshot_contact_state()
    assert len(first_records) == 1
    assert first_active == frozenset({"other-scenario"})

    recorder.clear_control_step()
    second_records, second_active = recorder.snapshot_contact_state()
    assert second_records == ()
    assert second_active == frozenset({"other-scenario"})

    PersistentEngine.physics_world.dynamic_world.manifolds = []
    recorder.clear_control_step()
    assert recorder.snapshot_contact_state() == ((), frozenset())


def test_metadrive_contact_recorder_ignores_unresolved_non_actor_contacts():
    class UnresolvedContact(_Contact):
        pass

    recorder = MetaDriveContactRecorder(_Env(), object_from_node=lambda _node: None)
    recorder.observe(UnresolvedContact())
    assert recorder.ignored_non_actor_contacts == 1
    assert recorder.snapshot_contact_state() == ((), frozenset())


def test_metadrive_contact_recorder_ignores_non_ego_actor_contacts():
    class ContactWithoutEgo(_Contact):
        def __init__(self):
            self.node0 = _Node(_OtherVehicle())
            self.node1 = _Node(_OtherVehicle())
            self.manifold_point = _Manifold()

    recorder = MetaDriveContactRecorder(_Env(), object_from_node=_object_from_node)
    recorder.observe(ContactWithoutEgo())
    assert recorder.ignored_non_actor_contacts == 1
    assert recorder.snapshot_contact_state() == ((), frozenset())


def test_metadrive_contact_recorder_defers_node_only_callback_to_manifold_query():
    class NodeOnlyContact(_Contact):
        getManifoldPoint = None

        def __init__(self):
            super().__init__()
            self.manifold_point = None

    recorder = MetaDriveContactRecorder(_Env(), object_from_node=_object_from_node)
    recorder.observe(NodeOnlyContact())
    assert recorder.deferred_manifold_contacts == 1
    assert recorder.snapshot_contact_state() == ((), frozenset())


def test_metadrive_contact_rejects_contacts_without_ego():
    class ContactWithoutEgo(_Contact):
        def __init__(self):
            self.node0 = _Node(_OtherVehicle())
            self.node1 = _Node(_OtherVehicle())
            self.manifold_point = _Manifold()

    with pytest.raises(ValueError, match="ego vehicle"):
        contact_onset_from_metadrive_contact(
            ContactWithoutEgo(), _Env(), object_from_node=_object_from_node
        )
