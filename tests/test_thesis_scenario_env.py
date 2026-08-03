from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from thesis_rl.envs.scene_context import SceneContextAdapter
from thesis_rl.envs import thesis_scenario_env as thesis_env_module
from thesis_rl.envs.thesis_scenario_env import scenario_time_limit_reached
from thesis_rl.mission.types import MissionSnapshot
from thesis_rl.runtime.wiring.builders import collect_scenario_runtime_stats


@dataclass(frozen=True)
class _MissionDoneSnapshot:
    step_index: int
    mission_snapshot: MissionSnapshot | None = None


@pytest.mark.parametrize(
    ("steps", "length", "extra", "expected"),
    [
        (9, 10, 0, False),
        (10, 10, 0, True),
        (59, 10, 50, False),
        (60, 10, 50, True),
    ],
)
def test_scenario_time_limit_supports_zero_and_tail(
    steps: int, length: int, extra: int, expected: bool
) -> None:
    assert (
        scenario_time_limit_reached(
            episode_steps=steps,
            scenario_length=length,
            extra_steps_after_scenario=extra,
        )
        is expected
    )


def test_scene_context_separates_line_from_physical_boundary() -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace(
        on_yellow_continuous_line=True,
        on_white_continuous_line=False,
        crash_sidewalk=False,
        navigation=SimpleNamespace(current_lateral=1.0, route_completion=0.4),
        dist_to_left_side=3.0,
        dist_to_right_side=3.0,
        LENGTH=4.5,
        WIDTH=1.9,
    )
    env = SimpleNamespace(config={"max_lateral_dist": 4.0})

    assert adapter.is_on_continuous_line(vehicle) is True
    assert adapter.is_physically_out_of_road(env, vehicle) is False
    assert adapter.get_ego_dimensions(vehicle) == (4.5, 1.9)

    # Route deviation is not physical road exit when both native surface
    # distances remain positive.
    vehicle.navigation.current_lateral = 5.0
    assert adapter.is_physically_out_of_road(env, vehicle) is False


def test_scene_context_does_not_treat_reference_lane_distance_as_road_exit() -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace(
        crash_sidewalk=False,
        navigation=SimpleNamespace(current_lateral=3.3987159729003906),
        dist_to_left_side=4.398715972900391,
        dist_to_right_side=-0.3987159729003906,
        on_lane=True,
    )
    env = SimpleNamespace(config={"max_lateral_dist": 4.0})

    assert adapter.is_physically_out_of_road(env, vehicle) is False


def test_scene_context_terminates_full_rulebook_geometric_exit_without_contact() -> None:
    adapter = SceneContextAdapter()
    ego = SimpleNamespace(
        footprint=box(10.0, 10.0, 12.0, 12.0),
        position_xy=(11.0, 11.0),
        position_z=0.0,
    )
    lane = SimpleNamespace(
        lane_id="lane-a",
        centerline=SimpleNamespace(project=lambda _position: SimpleNamespace(z_m=0.0)),
        polygon_xy=box(-2.0, -2.0, 2.0, 2.0),
    )
    env = SimpleNamespace(
        rulebook_v2_adapter=SimpleNamespace(
            snapshotter=lambda _env: SimpleNamespace(ego=ego),
            initial_cache=SimpleNamespace(route_lanes=(lane,)),
        )
    )
    vehicle = SimpleNamespace(crash_sidewalk=False, contact_results=())

    assert adapter.is_physically_out_of_road(env, vehicle) is True
    diagnostics = adapter.get_physical_road_diagnostics(env, vehicle)
    assert diagnostics["geometric_full_footprint_exit"] is True
    assert diagnostics["geometric_outside_area_m2"] == pytest.approx(4.0)
    assert diagnostics["geometric_ego_area_m2"] == pytest.approx(4.0)


def test_scene_context_does_not_terminate_partial_rulebook_geometric_exit() -> None:
    adapter = SceneContextAdapter()
    ego = SimpleNamespace(
        footprint=box(1.0, -1.0, 3.0, 1.0),
        position_xy=(2.0, 0.0),
        position_z=0.0,
    )
    lane = SimpleNamespace(
        lane_id="lane-a",
        centerline=SimpleNamespace(project=lambda _position: SimpleNamespace(z_m=0.0)),
        polygon_xy=box(-2.0, -2.0, 2.0, 2.0),
    )
    env = SimpleNamespace(
        rulebook_v2_adapter=SimpleNamespace(
            snapshotter=lambda _env: SimpleNamespace(ego=ego),
            initial_cache=SimpleNamespace(route_lanes=(lane,)),
        )
    )
    vehicle = SimpleNamespace(crash_sidewalk=False, contact_results=())

    assert adapter.is_physically_out_of_road(env, vehicle) is False
    diagnostics = adapter.get_physical_road_diagnostics(env, vehicle)
    assert diagnostics["geometric_full_footprint_exit"] is False
    assert diagnostics["geometric_outside_area_m2"] == pytest.approx(2.0)
    assert diagnostics["geometric_ego_area_m2"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("contacts", "expected"),
    [
        ({"ROAD_EDGE_BOUNDARY"}, False),
        ({"ROAD_EDGE_SIDEWALK"}, True),
        ({"GUARDRAIL"}, True),
    ],
)
def test_scene_context_does_not_treat_boundary_probe_as_sidewalk_exit(
    contacts: set[str], expected: bool
) -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace(
        crash_sidewalk=True,
        contact_results=contacts,
        navigation=SimpleNamespace(current_lateral=0.0),
        dist_to_left_side=1.0,
        dist_to_right_side=1.0,
    )
    env = SimpleNamespace(config={"max_lateral_dist": 4.0})

    assert adapter.is_physically_out_of_road(env, vehicle) is expected


def test_scene_context_termination_reason_is_stable() -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace()
    assert (
        adapter.get_termination_reason(
            None,
            vehicle,
            {"crash_vehicle": True, "max_step": True},
        )
        == "crash_vehicle"
    )


def test_scenario_time_limit_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        scenario_time_limit_reached(
            episode_steps=0,
            scenario_length=0,
            extra_steps_after_scenario=0,
        )


def test_thesis_success_rejects_short_reference_trajectory() -> None:
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.config = {
        "minimum_success_route_length_m": 10.0,
        "success_route_completion_threshold": 0.95,
    }
    env._mission_runtime = SimpleNamespace(
        snapshot=MissionSnapshot("mission", 0, 0, 5.0, 0.0, True, False, False)
    )
    vehicle = SimpleNamespace(
        navigation=SimpleNamespace(
            route_completion=0.03,
            reference_trajectory=SimpleNamespace(length=5.0),
        )
    )

    assert env._is_thesis_success(vehicle) is False


def test_thesis_route_completion_is_bounded_for_metrics() -> None:
    assert thesis_env_module.ThesisScenarioEnv._normalise_route_completion(1.25) == (1.0, 1.25)
    assert thesis_env_module.ThesisScenarioEnv._normalise_route_completion(-0.25) == (0.0, -0.25)


def test_assigned_route_metadata_is_published_to_reset_config() -> None:
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.config = {}
    env.current_scenario_record = SimpleNamespace(
        assigned_route_lane_ids=("lane-a", "lane-b"),
        assigned_route_source="pg_sdc_offline_task_annotation",
    )

    env._publish_assigned_route_metadata()

    assert env.config["assigned_route_lane_ids"] == ("lane-a", "lane-b")
    assert env.config["assigned_route_source"] == "pg_sdc_offline_task_annotation"


def test_assigned_route_metadata_supports_metadrive_style_single_argument_pop() -> None:
    """Regression for vectorized ScenarioEnv workers using MetaDrive Config."""

    class SingleArgumentPopConfig(dict[str, object]):
        def pop(self, key: str) -> object:  # type: ignore[override]
            return super().pop(key)

    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.config = SingleArgumentPopConfig(
        assigned_route_lane_ids=("stale-lane",),
        assigned_route_source="stale-source",
    )
    env.current_scenario_record = SimpleNamespace(
        assigned_route_lane_ids=("lane-a", "lane-b"),
        assigned_route_source="waymo_sdc_offline_task_annotation",
    )

    env._publish_assigned_route_metadata()

    assert env.config["assigned_route_lane_ids"] == ("lane-a", "lane-b")
    assert env.config["assigned_route_source"] == "waymo_sdc_offline_task_annotation"


def test_missing_assigned_route_fails_closed_before_reset() -> None:
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.config = {"assigned_route_lane_ids": ("stale-lane",)}
    env.current_scenario_record = SimpleNamespace(assigned_route_lane_ids=())

    with pytest.raises(ValueError, match="missing assigned_route_lane_ids"):
        env._publish_assigned_route_metadata()
    assert "assigned_route_lane_ids" not in env.config


def test_reset_injects_frozen_route_metadata_without_inspecting_track() -> None:
    env = SimpleNamespace(
        current_scenario_record=SimpleNamespace(
            assigned_route_lane_ids=("lane-a", "lane-b"),
            assigned_route_source="waymo_sdc_offline_task_annotation",
        ),
        engine=SimpleNamespace(
            data_manager=SimpleNamespace(current_scenario={"metadata": {}, "tracks": "sentinel"})
        ),
    )

    thesis_env_module.ThesisScenarioEnv._inject_assigned_route_metadata_into_scenario(env)

    assert env.engine.data_manager.current_scenario["metadata"] == {
        "assigned_route_lane_ids": ["lane-a", "lane-b"],
        "assigned_route_source": "waymo_sdc_offline_task_annotation",
    }
    assert env.engine.data_manager.current_scenario["tracks"] == "sentinel"


def test_provider_seed_selection_does_not_access_data_manager_before_reset() -> None:
    record = SimpleNamespace(
        scenario_uid="pg:scenario",
        runtime_index=17,
        source="pg",
        primary_arm="A0_simple_low_traffic",
        assigned_route_lane_ids=("lane-a",),
        assigned_route_source="pg_sdc_offline_task_annotation",
    )
    provider = SimpleNamespace(
        sample=lambda **_kwargs: record,
        sampling_metadata=lambda **_kwargs: {},
    )
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.scenario_provider = provider
    env.scenario_excluded_uids = set()
    env.split = "train"
    env.worker_id = 0
    env.scenario_arm = None
    env.current_scenario_record = None
    env.config = {}
    env.seed_calls = []
    env.seed = lambda seed: env.seed_calls.append(seed)
    env._inject_assigned_route_metadata_into_scenario = lambda: (_ for _ in ()).throw(
        AssertionError("route metadata must be injected after ScenarioDataManager.reset")
    )

    thesis_env_module.ThesisScenarioEnv._reset_global_seed(env)

    assert env.seed_calls == [17]
    assert env.current_scenario_record is record


def test_causal_builder_is_built_from_persisted_route_metadata() -> None:
    scenario = {
        "metadata": {
            "sdc_id": "ego",
            "assigned_route_lane_ids": ["lane-a"],
            "assigned_route_source": "waymo_sdc_offline_task_annotation",
        },
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1000.0, 1000.0, 0.0]],
                    "heading": [3.14],
                    "valid": [True],
                }
            }
        },
        "map_features": {
            "lane-a": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
                "width": [3.5, 3.5],
            }
        },
    }
    record = SimpleNamespace(source="waymo", scenario_uid="builder-route")

    builder = thesis_env_module.ThesisScenarioEnv._build_causal_frame_builder(scenario, record, {})

    assert builder.route_navigation.OUTPUT_DIM == 22
    assert builder.route_navigation.waypoint_adapter.route.length_m == 100.0


def test_causal_builder_is_installed_only_on_observations_that_request_it() -> None:
    observation = SimpleNamespace(builder=None)
    observation.set_frame_builder = lambda builder: setattr(observation, "builder", builder)
    env = SimpleNamespace(
        current_scenario_record=SimpleNamespace(source="waymo", scenario_uid="builder-route"),
        config={},
        agent_manager=SimpleNamespace(observations={"agent": observation}),
        engine=SimpleNamespace(
            data_manager=SimpleNamespace(
                current_scenario={
                    "metadata": {
                        "assigned_route_lane_ids": ["lane-a"],
                        "assigned_route_source": "waymo_sdc_offline_task_annotation",
                    },
                    "map_features": {
                        "lane-a": {
                            "type": "LANE_SURFACE_STREET",
                            "polyline": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
                            "width": [3.5, 3.5],
                        }
                    },
                }
            )
        ),
    )
    env._build_causal_frame_builder = (
        thesis_env_module.ThesisScenarioEnv._build_causal_frame_builder
    )

    thesis_env_module.ThesisScenarioEnv._install_causal_observation_builder(env)

    assert observation.builder is not None


def test_thesis_reward_suppresses_native_short_route_bonus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.config = {
        "minimum_success_route_length_m": 10.0,
        "success_route_completion_threshold": 0.95,
    }
    vehicle = SimpleNamespace(
        navigation=SimpleNamespace(
            route_completion=0.03,
            reference_trajectory=SimpleNamespace(length=1.5),
        )
    )
    env.agent_manager = SimpleNamespace(active_agents={"default_agent": vehicle})
    env._mission_runtime = SimpleNamespace(
        snapshot=MissionSnapshot("mission", 0, 0, 5.0, 0.0, True, False, False)
    )
    monkeypatch.setattr(
        thesis_env_module.ScenarioEnv,
        "reward_function",
        lambda _self, _vehicle_id: (5.0, {"step_reward": 0.12}),
    )
    monkeypatch.setattr(
        thesis_env_module.ThesisScenarioEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )

    reward, info = env.reward_function("default_agent")

    assert reward == pytest.approx(0.12)
    assert info["success_reward_suppressed"] is True
    assert info["thesis_success"] is False


def _make_done_test_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_done: bool,
    done_info: dict[str, bool],
    episode_steps: int = 1,
    scenario_length: int = 100,
    extra_steps: int = 50,
    lateral: float = 0.0,
    continuous_line: bool = False,
    route_completion: float = 0.4,
    reference_route_length: float | None = None,
):
    reference_trajectory = (
        SimpleNamespace(length=reference_route_length)
        if reference_route_length is not None
        else None
    )
    vehicle = SimpleNamespace(
        on_yellow_continuous_line=continuous_line,
        on_white_continuous_line=False,
        crash_sidewalk=False,
        navigation=SimpleNamespace(
            current_lateral=lateral,
            route_completion=route_completion,
            reference_trajectory=reference_trajectory,
        ),
        dist_to_left_side=1.0,
        dist_to_right_side=1.0,
        on_lane=True,
        LENGTH=4.5,
        WIDTH=1.9,
    )
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.agent_manager = SimpleNamespace(active_agents={"default_agent": vehicle})
    env.episode_lengths = {"default_agent": episode_steps}
    env.config = {"max_lateral_dist": 4.0, "extra_steps_after_scenario": extra_steps}
    fake_engine = SimpleNamespace(
        data_manager=SimpleNamespace(current_scenario_length=scenario_length)
    )
    monkeypatch.setattr(
        thesis_env_module.ThesisScenarioEnv,
        "engine",
        property(lambda _self: fake_engine),
    )
    env.scene_context = SceneContextAdapter()
    mission_snapshot = MissionSnapshot("mission", 0, 0, 5.0, 0.0, True, False, False)
    env._mission_runtime = SimpleNamespace(
        snapshot=mission_snapshot,
        update=lambda _pre, _post: mission_snapshot,
    )
    env._mission_pre_snapshot = _MissionDoneSnapshot(0, mission_snapshot)
    env._mission_snapshotter = lambda _env: _MissionDoneSnapshot(1, mission_snapshot)
    env._last_done_info = {}
    monkeypatch.setattr(
        thesis_env_module.ScenarioEnv,
        "done_function",
        lambda _self, _vehicle_id: (base_done, dict(done_info)),
    )
    return env


def test_thesis_done_makes_continuous_line_only_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"out_of_road": True},
        continuous_line=True,
    )

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["crossed_continuous_line"] is True
    assert info["physical_out_of_road"] is False
    assert info["out_of_road"] is False


def test_thesis_done_updates_mission_once_per_committed_episode_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(monkeypatch, base_done=False, done_info={}, episode_steps=0)
    reset_snapshot = MissionSnapshot("mission", 0, 0, 10.0, 0.0, True, False, False)
    committed_snapshot = MissionSnapshot("mission", 1, 0, 9.0, 0.1, True, False, False)
    updates: list[tuple[object, object]] = []

    class Runtime:
        snapshot = reset_snapshot

        def update(self, pre, post):
            updates.append((pre, post))
            self.snapshot = committed_snapshot
            return committed_snapshot

    env._mission_runtime = Runtime()
    env._mission_pre_snapshot = _MissionDoneSnapshot(0, reset_snapshot)
    env._mission_snapshotter = lambda _env: _MissionDoneSnapshot(1, reset_snapshot)

    env.done_function("default_agent")
    assert updates == []

    env.episode_lengths["default_agent"] = 1
    env.done_function("default_agent")
    env.done_function("default_agent")

    assert len(updates) == 1
    assert env._mission_pre_snapshot.mission_snapshot == committed_snapshot


def test_thesis_done_does_not_terminate_route_deviation_on_road(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"out_of_road": True},
        lateral=5.0,
    )

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["physical_out_of_road"] is False
    assert info["out_of_road"] is False
    assert info["route_lateral"] == 5.0
    assert info["dist_to_right_side"] == 1.0


def test_thesis_done_makes_boundary_probe_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"out_of_road": True, "crash_sidewalk": True, "crash": True},
    )
    vehicle = env.agent_manager.active_agents["default_agent"]
    vehicle.contact_results = {"ROAD_EDGE_BOUNDARY"}

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["out_of_road"] is False
    assert info["crash_sidewalk"] is False
    assert info["crash"] is False
    assert info["physical_out_of_road"] is False


def test_thesis_native_out_of_road_predicate_ignores_boundary_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=False,
        done_info={},
    )
    vehicle = env.agent_manager.active_agents["default_agent"]
    vehicle.crash_sidewalk = True
    vehicle.contact_results = {"ROAD_EDGE_BOUNDARY"}

    assert env._is_out_of_road(vehicle) is False


def test_thesis_done_preserves_physical_sidewalk_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"crash_sidewalk": True, "crash": True},
    )
    vehicle = env.agent_manager.active_agents["default_agent"]
    vehicle.crash_sidewalk = True
    vehicle.contact_results = {"ROAD_EDGE_SIDEWALK"}

    done, info = env.done_function("default_agent")

    assert done is True
    assert info["crash_sidewalk"] is True
    assert info["crash"] is True
    assert info["physical_out_of_road"] is True


def test_thesis_done_keeps_physical_exit_terminal_with_line_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"out_of_road": True},
        continuous_line=True,
    )
    vehicle = env.agent_manager.active_agents["default_agent"]
    vehicle.crash_sidewalk = True
    vehicle.contact_results = {"ROAD_EDGE_SIDEWALK"}

    done, info = env.done_function("default_agent")

    assert done is True
    assert info["crossed_continuous_line"] is True
    assert info["physical_out_of_road"] is True
    assert info["out_of_road"] is True


def test_thesis_done_does_not_terminate_live_regression_reference_lane_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"out_of_road": True},
    )
    vehicle = env.agent_manager.active_agents["default_agent"]
    vehicle.navigation.current_lateral = 3.3987159729003906
    vehicle.dist_to_left_side = 4.398715972900391
    vehicle.dist_to_right_side = -0.3987159729003906
    vehicle.contact_results = {"ROAD_LINE_BROKEN_SINGLE_WHITE"}

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["physical_out_of_road"] is False
    assert info["out_of_road"] is False
    assert info["contact_results"] == ["ROAD_LINE_BROKEN_SINGLE_WHITE"]


def test_thesis_done_preserves_native_collision_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"crash_vehicle": True},
    )

    done, info = env.done_function("default_agent")

    assert done is True
    assert info["termination_reason"] == "crash_vehicle"


def test_thesis_done_rejects_native_short_route_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"arrive_dest": True},
        route_completion=0.03,
        reference_route_length=5.0,
    )

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["arrive_dest"] is False
    assert info["termination_reason"] is None


def test_thesis_done_preserves_generic_collision_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=True,
        done_info={"crash": True},
    )

    done, info = env.done_function("default_agent")

    assert done is True
    assert info["termination_reason"] == "collision"


def test_thesis_done_adds_custom_timeout_without_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_done_test_env(
        monkeypatch,
        base_done=False,
        done_info={},
        episode_steps=10,
        scenario_length=10,
        extra_steps=0,
    )

    done, info = env.done_function("default_agent")

    assert done is False
    assert info["max_step"] is True
    assert info["termination_reason"] == "time_limit"


def test_collect_scenario_runtime_stats_merges_vector_workers() -> None:
    class FakeVectorEnv:
        def env_method(self, name: str):
            assert name == "get_runtime_stats"
            return [
                {
                    "resets": 2,
                    "steps": 4,
                    "episodes": 1,
                    "resets_by_source": {"waymo": 2},
                    "steps_by_source": {"waymo": 4},
                    "episodes_by_arm": {"A0": 1},
                    "resets_by_source_arm": {"waymo": {"A0": 2}},
                    "steps_by_source_arm": {"waymo": {"A0": 4}},
                    "episodes_by_source_arm": {"waymo": {"A0": 1}},
                    "termination_reasons": {"success": 1},
                },
                {
                    "resets": 1,
                    "steps": 3,
                    "episodes": 1,
                    "resets_by_source": {"pg": 1},
                    "steps_by_source": {"pg": 3},
                    "episodes_by_arm": {"A1": 1},
                    "resets_by_source_arm": {"pg": {"A1": 1}},
                    "steps_by_source_arm": {"pg": {"A1": 3}},
                    "episodes_by_source_arm": {"pg": {"A1": 1}},
                    "termination_reasons": {"truncated": 1},
                },
            ]

    stats = collect_scenario_runtime_stats(FakeVectorEnv())
    assert stats is not None
    assert stats["resets"] == 3
    assert stats["steps_by_source"] == {"waymo": 4, "pg": 3}
    assert stats["episodes_by_arm"] == {"A0": 1, "A1": 1}
    assert stats["steps_by_source_arm"] == {"waymo": {"A0": 4}, "pg": {"A1": 3}}
