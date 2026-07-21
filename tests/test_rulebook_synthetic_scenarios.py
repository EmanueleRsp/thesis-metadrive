from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact
from thesis_rl.rulebook.v2.context.live_adapter import LiveSnapshotAdapter, LiveSnapshotSources
from thesis_rl.rulebook.v2.context.metadrive_live import (
    live_actor_snapshots,
    live_ego_snapshot,
    live_signal_states_by_physical_id,
)
from thesis_rl.rulebook.v2.transition import (
    RulebookTransitionConfig,
    build_episode_cache,
    initial_memory_for_snapshot,
    transition_evaluator_factory,
)
from rulebook_scenario_fixtures import build_scenarios, write_persistent_fixtures


PERSISTED_FIXTURE_ROOT = Path("tests/fixtures/rulebook_scenarios")


def test_synthetic_descriptors_pass_metadrive_schema_validation() -> None:
    from metadrive.scenario.scenario_description import ScenarioDescription

    for scenario in build_scenarios().values():
        ScenarioDescription.sanity_check(scenario, check_self_type=True)


def test_synthetic_descriptors_normalize_with_frozen_assigned_route() -> None:
    for fixture_id, scenario in build_scenarios().items():
        result = build_waymo_static_adapter_result(scenario, scenario_uid=f"synthetic:{fixture_id}")
        assert result.task_route.lane_ids == ("lane-ego",)
        assert not result.validation_errors


def test_persistent_fixture_generation_is_reloadable(tmp_path: Path) -> None:
    manifest_path = write_persistent_fixtures(tmp_path)
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["fixtures"]] == [
        "red_light",
        "crosswalk_pedestrian",
        "vehicle_pedestrian_collision",
    ]
    for item in manifest["fixtures"]:
        with (tmp_path / item["file"]).open("rb") as handle:
            scenario = pickle.load(handle)
        assert scenario["id"] == item["scenario_id"]


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


def test_red_light_descriptor_resets_and_steps_in_scenario_online_env() -> None:
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    scenario = build_scenarios()["red_light"]
    env = ScenarioOnlineEnv(
        {
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": False,
            "no_light": False,
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


def test_red_light_descriptor_evaluates_one_live_rulebook_transition() -> None:
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    fixture_id = "red_light"
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
        pre_state = snapshotter.capture(env)
        _observation, _reward, _terminated, _truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        )
        post_state = snapshotter.capture(env)
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
        assert result.components["signal"].evaluable
        assert result.components["signal"].applicable
    finally:
        env.close()
