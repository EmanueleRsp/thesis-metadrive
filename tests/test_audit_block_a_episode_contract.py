"""Audit 2026-09-06, block A: the episode contract reads the Rulebook's snapshot.

A1: ``done_function`` must see the actor-complete snapshot the Rulebook v2
adapter installs, otherwise ADR-071's not-at-fault truncation can never fire.
A6: the at-fault classification in the env must read the same pre-transition
inputs as ``evaluate_collision_impact``, and the frontal band must be the ego's
width, not its length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, pi, sin
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

from thesis_rl.envs import thesis_scenario_env as thesis_env_module
from thesis_rl.envs.scene_context import SceneContextAdapter
from thesis_rl.mission.types import MissionSnapshot
from thesis_rl.rulebook.v2.components.collision_fault import CollisionFault, classify_contact
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


def _rectangle(center: tuple[float, float], heading: float, length: float, width: float):
    cx, cy = center
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x = sx * length / 2.0
        y = sy * width / 2.0
        corners.append(
            (cx + x * cos(heading) - y * sin(heading), cy + x * sin(heading) + y * cos(heading))
        )
    return Polygon(corners)


def _actor(
    actor_id: str,
    position: tuple[float, float],
    velocity: tuple[float, float],
    *,
    heading: float = 0.0,
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        ActorClass.VEHICLE,
        position,
        0.0,
        heading,
        velocity,
        _rectangle(position, heading, 4.8, 1.8),
        None,
        20.0,
    )


# --- A6, half width -------------------------------------------------------


def test_front_band_is_the_ego_half_width_not_half_length() -> None:
    """A moving agent 1.5 m to the side is a lateral impact: outside a 0.9 m
    half width, inside the 2.4 m half length the old code used."""

    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    side = _actor("side", (3.0, 1.5), (5.0, 0.0))
    assert classify_contact(pre_ego=ego, actor=side) is CollisionFault.ACTIVE_LATERAL

    ahead = _actor("ahead", (3.0, 0.5), (5.0, 0.0))
    assert classify_contact(pre_ego=ego, actor=ahead) is CollisionFault.ACTIVE_FRONT


def test_front_band_does_not_inflate_with_the_ego_heading() -> None:
    heading = pi / 4.0
    ego = _actor("ego", (0.0, 0.0), (5.0 * cos(heading), 5.0 * sin(heading)), heading=heading)
    # 3 m ahead along the heading, 1.5 m to the left of it.
    dx = 3.0 * cos(heading) - 1.5 * sin(heading)
    dy = 3.0 * sin(heading) + 1.5 * cos(heading)
    side = _actor("side", (dx, dy), (1.0, 1.0), heading=heading)
    assert classify_contact(pre_ego=ego, actor=side) is CollisionFault.ACTIVE_LATERAL


# --- A6, done_function reads the pre-state --------------------------------


@dataclass(frozen=True)
class _Snapshot:
    step_index: int
    ego: ActorSnapshot
    actors: tuple[ActorSnapshot, ...] = ()
    contact_onset_records: tuple[ContactOnsetRecord, ...] = ()
    mission_snapshot: MissionSnapshot | None = None


def _make_env(monkeypatch: pytest.MonkeyPatch, *, pre: _Snapshot, post: _Snapshot):
    vehicle = SimpleNamespace(
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
        crash_sidewalk=False,
        navigation=SimpleNamespace(
            current_lateral=0.0, route_completion=0.4, reference_trajectory=None
        ),
        dist_to_left_side=1.0,
        dist_to_right_side=1.0,
        on_lane=True,
        LENGTH=4.5,
        WIDTH=1.9,
    )
    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env.agent_manager = SimpleNamespace(active_agents={"default_agent": vehicle})
    env.episode_lengths = {"default_agent": 1}
    env.config = {"max_lateral_dist": 4.0, "extra_steps_after_scenario": 0}
    fake_engine = SimpleNamespace(data_manager=SimpleNamespace(current_scenario_length=100))
    monkeypatch.setattr(
        thesis_env_module.ThesisScenarioEnv, "engine", property(lambda _self: fake_engine)
    )
    env.scene_context = SceneContextAdapter()
    mission = MissionSnapshot("mission", 0, 0, 5.0, 0.0, True, False, False)
    env._mission_runtime = SimpleNamespace(snapshot=mission, update=lambda _pre, _post: mission)
    env._mission_pre_snapshot = replace(pre, mission_snapshot=mission)
    env._mission_snapshotter = lambda _env: replace(post, mission_snapshot=mission)
    env._mission_last_update_episode_length = None
    env._last_done_info = {}
    env.rulebook_v2_adapter = SimpleNamespace(initial_cache=SimpleNamespace(route_lanes=()))
    monkeypatch.setattr(
        thesis_env_module.ScenarioEnv,
        "done_function",
        lambda _self, _vehicle_id: (True, {"crash_vehicle": True, "crash": True}),
    )
    return env


def test_rear_impact_on_a_stopped_ego_truncates_instead_of_terminating(monkeypatch) -> None:
    """ADR-071 with real inputs: the stopped ego is struck from behind."""

    ego_pre = _actor("ego", (0.0, 0.0), (0.0, 0.0))
    other_pre = _actor("other", (-5.0, 0.0), (6.0, 0.0))
    pre = _Snapshot(0, ego_pre, (other_pre,))
    post = _Snapshot(
        1,
        ego_pre,
        (replace(other_pre, position_xy=(-4.6, 0.0)),),
        (ContactOnsetRecord("other", ActorClass.VEHICLE),),
    )
    env = _make_env(monkeypatch, pre=pre, post=post)

    done, info = env.done_function("default_agent")

    assert info["not_at_fault_collision"] is True
    assert done is False
    assert info["max_step"] is True
    assert info["crash_vehicle"] is False


def test_fault_is_classified_on_the_pre_state_like_the_rulebook(monkeypatch) -> None:
    """The other agent was moving in the pre-state and stopped in the post-state.

    The Rulebook charges from the pre-state (moving track behind the ego -> not
    at fault). Reading the post-state, as the env used to, would classify a
    STOPPED_TRACK and terminate while R1 charges nothing: the two would drift.
    """

    ego_pre = _actor("ego", (0.0, 0.0), (0.0, 0.0))
    other_pre = _actor("other", (-5.0, 0.0), (6.0, 0.0))
    other_post = replace(other_pre, position_xy=(-4.6, 0.0), velocity_xy=(0.0, 0.0))
    post = _Snapshot(
        1,
        replace(ego_pre, velocity_xy=(3.0, 0.0)),
        (other_post,),
        (ContactOnsetRecord("other", ActorClass.VEHICLE),),
    )
    env = _make_env(monkeypatch, pre=_Snapshot(0, ego_pre, (other_pre,)), post=post)

    done, info = env.done_function("default_agent")

    assert info["not_at_fault_collision"] is True
    assert done is False


def test_driving_into_a_stopped_track_still_terminates(monkeypatch) -> None:
    ego_pre = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other_pre = _actor("other", (5.0, 0.0), (0.0, 0.0))
    post = _Snapshot(1, ego_pre, (other_pre,), (ContactOnsetRecord("other", ActorClass.VEHICLE),))
    env = _make_env(monkeypatch, pre=_Snapshot(0, ego_pre, (other_pre,)), post=post)

    done, info = env.done_function("default_agent")

    assert info["not_at_fault_collision"] is False
    assert done is True
    assert info["crash_vehicle"] is True


def test_an_actor_unobserved_in_both_snapshots_terminates(monkeypatch) -> None:
    ego_pre = _actor("ego", (0.0, 0.0), (0.0, 0.0))
    post = _Snapshot(1, ego_pre, (), (ContactOnsetRecord("ghost", ActorClass.VEHICLE),))
    env = _make_env(monkeypatch, pre=_Snapshot(0, ego_pre, ()), post=post)

    done, info = env.done_function("default_agent")

    assert info["not_at_fault_collision"] is False
    assert done is True


# --- A1, the adapter installs the actor-complete snapshotter --------------


def test_rulebook_adapter_installation_replaces_the_actor_less_snapshotter(
    monkeypatch, tmp_path: Path
) -> None:
    """Before the fix `_mission_snapshotter` stayed the actor-less capture from
    `_install_mission_runtime`, so `contact_onset_records` was always empty in
    `done_function` and ADR-071 was dead code."""

    from thesis_rl.rulebook.v2 import transition as transition_module
    from thesis_rl.rulebook.v2 import wrapper as wrapper_module
    from thesis_rl.rulebook.v2.context import live_adapter as live_adapter_module
    from thesis_rl.rulebook.v2.context import metadrive_live as metadrive_live_module
    from thesis_rl.rulebook.v2.types import EnvSnapshot

    ego = _actor("ego", (0.0, 0.0), (0.0, 0.0))
    other = _actor("other", (-5.0, 0.0), (6.0, 0.0))
    onset = ContactOnsetRecord("other", ActorClass.VEHICLE)

    class _FakeLiveSnapshotAdapter:
        def __init__(self, _sources) -> None:
            pass

        def capture(self, env) -> EnvSnapshot:
            return EnvSnapshot(
                scenario_id="scenario",
                step_index=int(env.episode_step),
                sim_time_s=0.1 * int(env.episode_step),
                ego=ego,
                actors=(other,),
                contact_onset_records=(onset,),
                active_contact_ids=frozenset({"other"}),
                signal_states_by_physical_id={},
            )

    monkeypatch.setattr(live_adapter_module, "LiveSnapshotAdapter", _FakeLiveSnapshotAdapter)
    monkeypatch.setattr(
        live_adapter_module, "install_collision_callback_hook", lambda *a, **k: None
    )
    monkeypatch.setattr(
        metadrive_live_module,
        "MetaDriveContactRecorder",
        lambda *a, **k: SimpleNamespace(
            observe=lambda *_: None, snapshot_contact_state=lambda: ((), frozenset())
        ),
    )
    monkeypatch.setattr(transition_module, "initial_memory_for_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(transition_module, "transition_evaluator_factory", lambda *_: None)
    monkeypatch.setattr(
        wrapper_module, "RulebookV2Adapter", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    env = object.__new__(thesis_env_module.ThesisScenarioEnv)
    env._rulebook_v2_requested = True
    monkeypatch.setattr(
        thesis_env_module.ThesisScenarioEnv, "episode_step", property(lambda _self: 0)
    )
    env.config = {
        "data_directory": str(tmp_path / "runtime" / "train"),
        "decision_repeat": 5,
        "physics_world_step_size": 0.02,
    }
    fake_engine = SimpleNamespace(
        data_manager=SimpleNamespace(current_scenario={"id": "scenario"}),
        physics_world=SimpleNamespace(dynamic_world=object()),
    )
    monkeypatch.setattr(
        thesis_env_module.ThesisScenarioEnv, "engine", property(lambda _self: fake_engine)
    )
    env.current_scenario_record = SimpleNamespace(scenario_uid="waymo:fixture")
    env._mission_static_result = object()
    env._mission_episode_cache = SimpleNamespace(route_lanes=())
    mission = MissionSnapshot("mission", 0, 0, 5.0, 0.0, True, False, False)
    env._mission_runtime = SimpleNamespace(snapshot=mission, route=object())
    env._mission_pre_snapshot = _Snapshot(0, ego, mission_snapshot=mission)
    env._mission_snapshotter = lambda _env: _Snapshot(0, ego, mission_snapshot=mission)

    env._install_rulebook_v2_adapter()

    installed = env._mission_snapshotter(env)
    assert installed.contact_onset_records == (onset,)
    assert tuple(actor.actor_id for actor in installed.actors) == ("other",)
    assert installed.mission_snapshot is mission
    assert env._mission_pre_snapshot.actors == (other,)
    assert env.rulebook_v2_adapter.snapshotter(env).actors == (other,)
