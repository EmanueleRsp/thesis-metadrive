from __future__ import annotations

from math import cos, sin
from types import SimpleNamespace

from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import Vec3
from metadrive import MetaDriveEnv
from metadrive.constants import CollisionGroup
from metadrive.component.static_object.traffic_object import TrafficBarrier
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.component.vehicle.vehicle_type import DefaultVehicle

from thesis_rl.envs.observations.perception import (
    FirstHitLidarAdapter,
    SymbolicSignalVisibilityAdapter,
)


def test_pg_first_hit_lidar_resolves_static_barrier_and_hides_far_barrier() -> None:
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "use_render": False,
            "image_observation": False,
        }
    )
    try:
        env.reset(seed=0)
        ego = env.agent
        position = ego.position
        heading = ego.heading_theta
        near = env.engine.spawn_object(
            TrafficBarrier,
            position=(position[0] + 8.0 * cos(heading), position[1] + 8.0 * sin(heading)),
            heading_theta=heading,
            static=True,
        )
        far = env.engine.spawn_object(
            TrafficBarrier,
            position=(position[0] + 16.0 * cos(heading), position[1] + 16.0 * sin(heading)),
            heading_theta=heading,
            static=True,
        )

        sweep = FirstHitLidarAdapter().sweep(ego)

        assert sweep.beam_count == 240
        assert sweep.hit_count > 0
        assert near.id in sweep.actor_ids
        assert far.id not in sweep.actor_ids
    finally:
        env.close()


def test_pg_first_hit_lidar_resolves_pedestrian_vru() -> None:
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "use_render": False,
            "image_observation": False,
        }
    )
    try:
        env.reset(seed=0)
        ego = env.agent
        position = ego.position
        heading = ego.heading_theta
        pedestrian = env.engine.spawn_object(
            Pedestrian,
            position=(position[0] + 8.0 * cos(heading), position[1] + 8.0 * sin(heading)),
            heading_theta=heading,
            random_seed=1,
        )

        sweep = FirstHitLidarAdapter().sweep(ego)

        assert pedestrian.id in sweep.actor_ids
    finally:
        env.close()


def test_waymo_signal_mapping_supports_symbolic_visibility_gate_without_rgb() -> None:
    env = ScenarioEnv(
        {
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": True,
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    try:
        env.reset(seed=0)
        physical_ids = tuple(env.engine.light_manager._scenario_id_to_obj_id)
        visible = SymbolicSignalVisibilityAdapter().visible_ids(env.agent, physical_ids)

        assert physical_ids
        assert set(visible) == set(physical_ids)
        assert all(isinstance(value, bool) for value in visible.values())
    finally:
        env.close()


def test_waymo_signal_anchor_ray_accepts_a_controlled_visible_pose() -> None:
    env = ScenarioEnv(
        {
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": True,
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    try:
        env.reset(seed=0)
        physical_id, object_id = next(iter(env.engine.light_manager._scenario_id_to_obj_id.items()))
        light = env.engine.light_manager.spawned_objects[object_id]
        ego = env.agent
        ego.set_position((light.position[0] - 30.0, light.position[1]), height=0.0)
        ego.set_heading_theta(0.0)

        visible = SymbolicSignalVisibilityAdapter().visible_ids(ego, (physical_id,))

        assert visible == {physical_id: True}
    finally:
        env.close()


def test_waymo_signal_anchor_ray_is_blocked_by_a_physical_static_barrier() -> None:
    env = ScenarioEnv(
        {
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": True,
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    try:
        env.reset(seed=0)
        physical_id, object_id = next(iter(env.engine.light_manager._scenario_id_to_obj_id.items()))
        light = env.engine.light_manager.spawned_objects[object_id]
        ego = env.agent
        ego.set_position((light.position[0] - 10.0, light.position[1]), height=0.0)
        ego.set_heading_theta(0.0)
        adapter = SymbolicSignalVisibilityAdapter()
        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: True}

        env.engine.spawn_object(
            TrafficBarrier,
            position=(light.position[0] - 7.0, light.position[1]),
            heading_theta=0.0,
            static=True,
        )

        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: False}
    finally:
        env.close()


def test_waymo_signal_anchor_ray_is_blocked_by_a_physical_vehicle() -> None:
    env = ScenarioEnv(
        {
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": True,
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    try:
        env.reset(seed=0)
        physical_id, object_id = next(iter(env.engine.light_manager._scenario_id_to_obj_id.items()))
        light = env.engine.light_manager.spawned_objects[object_id]
        ego = env.agent
        ego.set_position((light.position[0] - 30.0, light.position[1]), height=0.0)
        ego.set_heading_theta(0.0)
        adapter = SymbolicSignalVisibilityAdapter()
        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: True}

        env.engine.spawn_object(
            DefaultVehicle,
            vehicle_config=env.config["vehicle_config"],
            position=(light.position[0] - 27.0, light.position[1]),
            heading=0.0,
        )

        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: False}
    finally:
        env.close()


def test_symbolic_signal_gate_rejects_out_of_fov_range_and_occlusion(monkeypatch) -> None:
    adapter = SymbolicSignalVisibilityAdapter()
    vehicle = SimpleNamespace(position=(0.0, 0.0), heading_theta=0.0)
    light = SimpleNamespace(position=(10.0, 0.0), origin=SimpleNamespace(getZ=lambda: 0.0))
    monkeypatch.setattr(adapter, "_has_blocker", lambda *_args, **_kwargs: False)

    assert adapter._is_visible(vehicle, light) is True
    light.position = (10.0, 10.0)
    assert adapter._is_visible(vehicle, light) is False
    light.position = (81.0, 0.0)
    assert adapter._is_visible(vehicle, light) is False
    light.position = (10.0, 0.0)
    monkeypatch.setattr(adapter, "_has_blocker", lambda *_args, **_kwargs: True)
    assert adapter._is_visible(vehicle, light) is False


def test_waymo_signal_anchor_ray_rejects_a_physical_occluder() -> None:
    env = ScenarioEnv(
        {
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": True,
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    node = None
    node_path = None
    try:
        env.reset(seed=0)
        physical_id, object_id = next(iter(env.engine.light_manager._scenario_id_to_obj_id.items()))
        light = env.engine.light_manager.spawned_objects[object_id]
        ego = env.agent
        ego.set_position((light.position[0] - 10.0, light.position[1]), height=0.0)
        ego.set_heading_theta(0.0)
        adapter = SymbolicSignalVisibilityAdapter()
        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: True}

        node = BulletRigidBodyNode("signal-occluder")
        node.addShape(BulletBoxShape(Vec3(0.5, 2.0, 4.0)))
        node.setIntoCollideMask(CollisionGroup.Vehicle)
        node.setStatic(True)
        node_path = env.engine.render.attachNewNode(node)
        node_path.setPos(light.position[0] - 5.0, light.position[1], 2.0)
        env.engine.physics_world.dynamic_world.attach(node)

        assert adapter.visible_ids(ego, (physical_id,)) == {physical_id: False}
    finally:
        if node is not None:
            env.engine.physics_world.dynamic_world.remove(node)
        if node_path is not None:
            node_path.removeNode()
        env.close()
