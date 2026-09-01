from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thesis_rl.mission.types import MissionSnapshot
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact
from thesis_rl.rulebook.v2.context.live_adapter import LiveSnapshotAdapter, LiveSnapshotSources
from thesis_rl.rulebook.v2.context.metadrive_live import (
    MetaDriveContactRecorder,
    live_actor_snapshots,
    live_ego_snapshot,
    live_signal_states_by_physical_id,
)
from thesis_rl.rulebook.v2.context.live_adapter import install_collision_callback_hook
from thesis_rl.rulebook.v2.memory import apply_cache_delta
from thesis_rl.rulebook.v2.transition import (
    RulebookTransitionConfig,
    build_episode_cache,
    initial_memory_for_snapshot,
    transition_evaluator_factory,
)
from thesis_rl.rulebook.v2.types import EnvSnapshot
from rulebook_scenario_fixtures import (
    FIXTURE_COMPONENTS,
    build_scenarios,
    write_persistent_fixtures,
)


PERSISTED_FIXTURE_ROOT = Path("tests/fixtures/rulebook_scenarios")


def _attach_synthetic_mission_snapshot(
    state: EnvSnapshot, route, *, previous_s_m: float | None = None
) -> tuple[EnvSnapshot, float]:
    """Attach a route-consistent ``MissionSnapshot`` to a live-captured snapshot.

    ``LiveSnapshotAdapter`` deliberately never populates ``mission_snapshot``
    (production wires it in afterwards from the real mission runtime, see
    ``ThesisScenarioEnv``'s ``replace(snapshot, mission_snapshot=...)``
    call sites). These synthetic-scenario tests exercise the raw
    ``ScenarioOnlineEnv`` without a mission runtime, so this mirrors that
    production pattern with an ``s_m`` projected onto the same
    ``EpisodeCache`` route the transition evaluator itself uses, satisfying
    R4's exact-canonical-route-station requirement (``AC-RCM-004``).
    """

    projection = route.project(state.ego.position_xy, previous_s_m=previous_s_m)
    completion = (
        min(max(projection.s_m / route.length_m, 0.0), 1.0) if route.length_m > 0.0 else 0.0
    )
    mission_snapshot = MissionSnapshot(
        "synthetic-test-mission",
        state.step_index,
        0,
        max(route.length_m - projection.s_m, 0.0),
        completion,
        True,
        False,
        False,
        s_m=projection.s_m,
    )
    return replace(state, mission_snapshot=mission_snapshot), projection.s_m


def test_synthetic_descriptors_pass_metadrive_schema_validation() -> None:
    from metadrive.scenario.scenario_description import ScenarioDescription

    for scenario in build_scenarios().values():
        ScenarioDescription.sanity_check(scenario, check_self_type=True)


def test_synthetic_descriptors_normalize_with_frozen_assigned_route() -> None:
    for fixture_id, scenario in build_scenarios().items():
        result = build_waymo_static_adapter_result(scenario, scenario_uid=f"synthetic:{fixture_id}")
        assert result.task_route.lane_ids == ("lane-ego",)
        if fixture_id == "unknown_signal":
            assert result.validation_errors == ("signal_state_unknown:lane-ego",)
        else:
            assert not result.validation_errors


def test_persistent_fixture_generation_is_reloadable(tmp_path: Path) -> None:
    manifest_path = write_persistent_fixtures(tmp_path)
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["fixtures"]] == list(build_scenarios())
    for item in manifest["fixtures"]:
        with (tmp_path / item["file"]).open("rb") as handle:
            scenario = pickle.load(handle)
        assert scenario["id"] == item["scenario_id"]


def test_synthetic_descriptor_generator_is_byte_deterministic() -> None:
    first = build_scenarios()
    second = build_scenarios()
    assert tuple(first) == tuple(second)
    for fixture_id in first:
        assert pickle.dumps(first[fixture_id], protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
            second[fixture_id], protocol=pickle.HIGHEST_PROTOCOL
        )


def test_checked_in_descriptors_match_the_versioned_generator() -> None:
    manifest_path = PERSISTED_FIXTURE_ROOT / "manifest.json"
    assert manifest_path.is_file(), "Run tests/generate_rulebook_scenarios.py before committing."
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "rulebook-synthetic-scenarios-v1"
    generated = build_scenarios()
    assert {item["id"] for item in manifest["fixtures"]} == set(generated)
    for item in manifest["fixtures"]:
        with (PERSISTED_FIXTURE_ROOT / item["file"]).open("rb") as handle:
            checked_in = pickle.load(handle)
        assert checked_in["id"] == generated[item["id"]]["id"]
        np.testing.assert_array_equal(
            checked_in["tracks"]["ego"]["state"]["position"],
            generated[item["id"]]["tracks"]["ego"]["state"]["position"],
        )
        assert item["components"] == list(FIXTURE_COMPONENTS[item["id"]])


@pytest.mark.parametrize("fixture_id", tuple(build_scenarios()))
def test_descriptors_reset_and_step_in_scenario_online_env(fixture_id: str) -> None:
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    scenario = build_scenarios()[fixture_id]
    env = ScenarioOnlineEnv(
        {
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": False,
            "no_light": False,
            "filter_overlapping_car": False,
        }
    )
    try:
        env.set_scenario(ScenarioDescription(scenario))
        observation, _ = env.reset(seed=0)
        assert np.isfinite(np.asarray(observation)).all()
        observation, reward, _terminated, _truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        )
        assert np.isfinite(np.asarray(observation)).all()
        assert np.isfinite(float(reward))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("fixture_id", "component_name", "expected_applicable", "expect_positive_cost"),
    (
        ("red_light", "signal", True, False),
        ("yellow_light", "signal", True, False),
        ("crosswalk_pedestrian", "crosswalk", True, True),
        ("vehicle_pedestrian_collision", "collision", False, False),
        ("vehicle_vehicle_collision", "collision", False, False),
        ("rss_front_vehicle", "rss", True, False),
        ("rss_front_vehicle", "ttc", True, False),
        # ADR-035 / DEV-EF-01: a same-lane leader is no longer a lateral-RSS
        # pair.  Two vehicles in one lane have a lateral gap of exactly zero
        # against d_safe^lat ~= 0.1625 m, so this scenario used to report
        # q_RSS,lat = 1.0 and mask the graded q_RSS,long.  The pair is now
        # NOT_APPLICABLE and stays covered by "rss" and "ttc" above.
        ("rss_front_vehicle", "rss_lateral", False, False),
        ("rss_rear_vehicle", "rss", False, False),
        ("rss_rear_vehicle", "rss_lateral", False, False),
        ("stop_sign", "stop", True, False),
        ("red_light", "progress", True, False),
        # No row for the `wrong_way` scene. ADR-066 deleted the rule -- one
        # violated step in 217,189 of expert replay -- so no live transition
        # produces a component for it. Retargeting the row onto
        # `wrong_carriageway` was tried and reverted: that scene drives the
        # wrong way along its own carriageway rather than entering the opposing
        # one, so the sub-rule is correctly NOT_APPLICABLE there and the row
        # would have asserted a violation the geometry does not contain. The
        # pure evaluator keeps its unit tests in `test_rulebook_v2_road.py`;
        # deregistration is asserted by
        # `test_rulebook_v51_levels.py::test_wrongway_is_deleted`.
        ("offroad", "offroad", True, True),
        ("solid_line", "solid_line", True, True),
        ("dashed_line", "dashed_line", True, False),
        ("vehicle_yield_pairwise", "vehicle_yield", True, False),
        ("vehicle_yield_stop", "vehicle_yield", True, False),
        ("vehicle_yield_roundabout", "vehicle_yield", True, False),
        ("vehicle_yield_occupied", "vehicle_yield", True, False),
    ),
)
def test_descriptors_evaluate_one_live_rulebook_transition(
    fixture_id: str,
    component_name: str,
    expected_applicable: bool,
    expect_positive_cost: bool,
) -> None:
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    scenario_uid = f"synthetic:{fixture_id}"
    scenario = build_scenarios()[fixture_id]
    static = build_waymo_static_adapter_result(scenario, scenario_uid=scenario_uid)
    cache = build_episode_cache(static)
    env = ScenarioOnlineEnv(
        {
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": False,
            "no_light": False,
            "filter_overlapping_car": False,
        }
    )
    try:
        env.set_scenario(ScenarioDescription(scenario))
        env.reset(seed=0)
        snapshotter = LiveSnapshotAdapter(
            LiveSnapshotSources(
                scenario_id=lambda _env: scenario_uid,
                step_index=lambda active_env: int(active_env.episode_step),
                sim_time_s=lambda active_env: float(active_env.episode_step * 0.1),
                ego=live_ego_snapshot,
                actors=live_actor_snapshots,
                contact_onset_records=lambda _env: (),
                active_contact_ids=lambda _env: frozenset(),
                signal_states_by_physical_id=live_signal_states_by_physical_id,
            )
        )
        pre_state, pre_s_m = _attach_synthetic_mission_snapshot(
            snapshotter.capture(env), cache.route_polyline
        )
        _observation, _reward, _terminated, _truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        )
        post_state, _post_s_m = _attach_synthetic_mission_snapshot(
            snapshotter.capture(env), cache.route_polyline, previous_s_m=pre_s_m
        )
        result, _next_memory, _cache_delta = transition_evaluator_factory(
            RulebookTransitionConfig(
                rss_calibration=RSSCalibrationArtifact(
                    config_hash="synthetic-test", ego_min_brake_mps2=4.0
                ),
                expected_config_hash="synthetic-test",
            )
        )(
            pre_state=pre_state,
            post_state=post_state,
            memory=initial_memory_for_snapshot(pre_state, cache),
            cache=cache,
        )
        assert result.complete_evaluation
        # TEST-EF-22 / REQ-EF-15 (M6a): the route-adherence diagnostics must be
        # wired through the live transition, not just computable in isolation.
        # They are diagnostic only and must never move a cost or a margin.
        progress = result.components["progress"]
        assert "route_outside_fraction" in progress.diagnostics
        outside = progress.diagnostics["route_outside_fraction"]
        assert 0.0 <= outside <= 1.0
        assert progress.diagnostics["route_adherence"] == pytest.approx(1.0 - outside)
        component = result.components[component_name]
        assert component.evaluable
        assert component.applicable is expected_applicable
        if expect_positive_cost:
            assert component.cost > 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    "fixture_id",
    (
        "vehicle_pedestrian_collision",
        "vehicle_cyclist_collision",
    ),
)
def test_vru_collision_descriptors_produce_a_live_collision_onset(fixture_id: str) -> None:
    from metadrive.engine.core.collision_callback import collision_callback
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription
    from metadrive.utils.utils import get_object_from_node

    scenario_uid = f"synthetic:{fixture_id}"
    scenario = build_scenarios()[fixture_id]
    cache = build_episode_cache(
        build_waymo_static_adapter_result(scenario, scenario_uid=scenario_uid)
    )
    env = ScenarioOnlineEnv(
        {
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": False,
            "no_light": False,
            "filter_overlapping_car": False,
        }
    )
    try:
        env.set_scenario(ScenarioDescription(scenario))
        env.reset(seed=0)
        recorder = MetaDriveContactRecorder(env, object_from_node=get_object_from_node)
        install_collision_callback_hook(
            env.engine.physics_world.dynamic_world,
            original_callback=collision_callback,
            observer=recorder.observe,
        )
        snapshotter = LiveSnapshotAdapter(
            LiveSnapshotSources(
                scenario_id=lambda _env: scenario_uid,
                step_index=lambda active_env: int(active_env.episode_step),
                sim_time_s=lambda active_env: float(active_env.episode_step * 0.1),
                ego=live_ego_snapshot,
                actors=live_actor_snapshots,
                contact_onset_records=lambda _env: recorder.snapshot_contact_state()[0],
                active_contact_ids=lambda _env: recorder.snapshot_contact_state()[1],
                signal_states_by_physical_id=live_signal_states_by_physical_id,
            )
        )
        pre_state, pre_s_m = _attach_synthetic_mission_snapshot(
            snapshotter.capture(env), cache.route_polyline
        )
        memory = initial_memory_for_snapshot(pre_state, cache)
        evaluator = transition_evaluator_factory(
            RulebookTransitionConfig(
                rss_calibration=RSSCalibrationArtifact(
                    config_hash="synthetic-test", ego_min_brake_mps2=4.0
                ),
                expected_config_hash="synthetic-test",
            )
        )
        observed_collision_costs: list[float] = []
        for _ in range(3):
            env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
            post_state, post_s_m = _attach_synthetic_mission_snapshot(
                snapshotter.capture(env), cache.route_polyline, previous_s_m=pre_s_m
            )
            result, memory, cache_delta = evaluator(
                pre_state=pre_state,
                post_state=post_state,
                memory=memory,
                cache=cache,
            )
            cache = apply_cache_delta(cache, cache_delta)
            observed_collision_costs.append(result.components["collision"].cost)
            pre_state = post_state
            pre_s_m = post_s_m
        assert any(cost > 0.0 for cost in observed_collision_costs)
    finally:
        env.close()
