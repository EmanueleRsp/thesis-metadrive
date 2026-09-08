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


class ScenarioTrafficLight:
    id = "traffic-light-runtime"


class _Node:
    def __init__(self, obj):
        self.obj = obj


class _Manifold:
    def getPositionWorldOnA(self):
        return (0.0, 0.0, 0.0)

    def getPositionWorldOnB(self):
        return (2.0, 0.0, 0.0)

    def getNormalWorldOnB(self):
        # Bullet convention: B -> A.  Here A is ego and B is the other actor.
        return (-1.0, 0.0, 0.0)


class _CoincidentManifold:
    def getPositionWorldOnA(self):
        return (0.0, 0.0, 0.0)

    def getPositionWorldOnB(self):
        return (0.0, 0.0, 0.0)

    def getNormalWorldOnB(self):
        # Bullet convention: B -> A.  For ego=A and other=B this is -X.
        return (-1.0, 0.0, 0.0)


class _StaleManifold(_CoincidentManifold):
    def getDistance(self):
        return 0.1


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


@pytest.fixture(autouse=True)
def _contain_traffic_scenario_mapping():
    """Restore `_Traffic.obj_id_to_scenario_id` after each test in this module.

    It is a *class* attribute holding one declared entry, and two tests below add
    `other-runtime -> other-scenario` to it without removing it, so every test
    running later in the same process inherited the addition. That is how
    `test_metadrive_contact_recorder_accepts_node_only_callback_without_manifold_query`
    came to pass — not because of what it asserts, but because a sibling had
    seeded the mapping it reads. Run on its own it failed, and it failed again
    under `pytest-xdist`, which hands tests of one file to different workers.

    Containing the mutation here closes the whole class of leak rather than the
    one instance: a test added later cannot silently depend on, or silently
    provide, an entry in this mapping. Only this mapping is restored — nothing
    else on `_Traffic` is mutated by any test in the module.
    """

    declared = dict(_Traffic.obj_id_to_scenario_id)
    yield
    _Traffic.obj_id_to_scenario_id.clear()
    _Traffic.obj_id_to_scenario_id.update(declared)


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


def test_metadrive_stationary_vehicle_stays_classified_as_vehicle():
    """REQ-R2-06 (rulebook v4.8): a parked/stationary VEHICLE-typed object
    must never be reclassified STATIC_COLLIDABLE based on its velocity, so
    it remains a candidate for the scoped lateral-RSS metric instead of
    silently falling out of R2 coverage entirely."""

    class ParkedVehicle(_Vehicle):
        id = "parked-runtime"
        velocity = (0.0, 0.0)

    snapshot = actor_snapshot_from_metadrive(_Env(), ParkedVehicle())
    assert snapshot.actor_class is ActorClass.VEHICLE


def test_metadrive_actor_requires_vehicle_speed_cap():
    class NoCapVehicle(_Vehicle):
        max_speed_m_s = None

    with pytest.raises(ValueError, match="speed cap"):
        actor_snapshot_from_metadrive(_Env(), NoCapVehicle())


def test_metadrive_contact_identifies_ego_and_other_without_manifold_geometry():
    env = _Env()
    _Traffic.obj_id_to_scenario_id["other-runtime"] = "other-scenario"
    record = contact_onset_from_metadrive_contact(
        _Contact(), env, object_from_node=_object_from_node
    )
    assert record.actor_id == "other-scenario"
    assert record.actor_class is ActorClass.VEHICLE


def test_metadrive_contact_uses_scenario_env_agent_as_ego():
    class AgentOwnedEnv:
        engine = type("Engine", (), {"traffic_manager": type("Traffic", (), {})()})()
        agents = {"ego": _Vehicle()}

    record = contact_onset_from_metadrive_contact(
        _Contact(), AgentOwnedEnv(), object_from_node=_object_from_node
    )
    assert record.actor_id == "other-runtime"


def test_metadrive_contact_recorder_clears_per_control_step():
    # Same precondition, established locally rather than inherited.
    _Traffic.obj_id_to_scenario_id["other-runtime"] = "other-scenario"

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


def test_metadrive_contact_recorder_does_not_keep_callback_only_contact_active():
    class World:
        def get_manifolds(self):
            return ()

    class Physics:
        dynamic_world = World()

    class QueryEngine:
        traffic_manager = _Traffic()
        physics_world = Physics()

    class QueryEnv:
        engine = QueryEngine()

    recorder = MetaDriveContactRecorder(QueryEnv(), object_from_node=_object_from_node)
    recorder.observe(_Contact())
    records, active = recorder.snapshot_contact_state()
    assert len(records) == 1
    assert active == frozenset()


def test_metadrive_contact_recorder_ignores_positive_distance_manifold():
    class World:
        manifolds = [_Contact()]

        def get_manifolds(self):
            return tuple(self.manifolds)

    class Physics:
        dynamic_world = World()

    class StaleContact(_Contact):
        def __init__(self):
            super().__init__()
            self.manifold_point = _StaleManifold()

    World.manifolds = [StaleContact()]

    class QueryEngine:
        traffic_manager = _Traffic()
        physics_world = Physics()

    class QueryEnv:
        engine = QueryEngine()

    recorder = MetaDriveContactRecorder(QueryEnv(), object_from_node=_object_from_node)
    records, active = recorder.snapshot_contact_state()
    assert records == ()
    assert active == frozenset()


def test_metadrive_contact_recorder_reads_base_vehicle_contact_test_path():
    class ContactResult:
        def getContacts(self):
            return (_Contact(),)

    class World:
        def get_manifolds(self):
            return ()

        def contactTest(self, node, use_filter):
            assert node == "ego-chassis"
            assert use_filter is True
            return ContactResult()

    class Chassis:
        def node(self):
            return "ego-chassis"

    class AgentOwnedVehicle(_Vehicle):
        chassis = Chassis()

    class ContactTestEnv:
        agents = {"ego": AgentOwnedVehicle()}
        engine = type(
            "Engine",
            (),
            {
                "traffic_manager": type("Traffic", (), {})(),
                "physics_world": type(
                    "Physics", (), {"static_world": World(), "dynamic_world": World()}
                )(),
            },
        )()

    recorder = MetaDriveContactRecorder(ContactTestEnv(), object_from_node=_object_from_node)
    records, active = recorder.snapshot_contact_state()
    assert len(records) == 1
    assert active == frozenset({"other-runtime"})


def test_metadrive_contact_test_ignores_lane_surface_contacts():
    class LaneNode(_Node):
        def getName(self):
            return "LANE_SURFACE_STREET"

    class LaneContact(_Contact):
        def __init__(self):
            self.node0 = LaneNode(_Vehicle())
            self.node1 = LaneNode(_OtherVehicle())
            self.manifold_point = _Manifold()

    class ContactResult:
        def getContacts(self):
            return (LaneContact(),)

    class World:
        def get_manifolds(self):
            return ()

        def contactTest(self, node, use_filter):
            return ContactResult()

    class Chassis:
        def node(self):
            return "ego-chassis"

    class ContactTestEnv:
        agents = {"ego": type("Vehicle", (), {"chassis": Chassis()})()}
        engine = type(
            "Engine",
            (),
            {"physics_world": type("Physics", (), {"static_world": World(), "dynamic_world": World()})()},
        )()

    recorder = MetaDriveContactRecorder(ContactTestEnv(), object_from_node=_object_from_node)
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


def test_metadrive_contact_recorder_accepts_node_only_callback_without_manifold_query():
    class NodeOnlyContact(_Contact):
        getManifoldPoint = None

        def __init__(self):
            super().__init__()
            self.manifold_point = None

    # Establish the runtime-to-scenario mapping this test reads, instead of
    # inheriting it from whichever sibling happened to run first.
    _Traffic.obj_id_to_scenario_id["other-runtime"] = "other-scenario"

    recorder = MetaDriveContactRecorder(_Env(), object_from_node=_object_from_node)
    recorder.observe(NodeOnlyContact())
    records, active = recorder.snapshot_contact_state()
    assert len(records) == 1
    assert records[0].actor_id == "other-scenario"
    assert active == frozenset({"other-scenario"})


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


def test_metadrive_contact_classifies_scenario_traffic_light_as_static_collision():
    class TrafficLightContact(_Contact):
        def __init__(self):
            self.node0 = _Node(_Vehicle())
            self.node1 = _Node(ScenarioTrafficLight())
            self.manifold_point = _Manifold()

    record = contact_onset_from_metadrive_contact(
        TrafficLightContact(), _Env(), object_from_node=_object_from_node
    )
    assert record.actor_id == "traffic-light-runtime"
    assert record.actor_class is ActorClass.STATIC_COLLIDABLE
