from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.runtime.io.video_diagnostics import (
    DiagnosticState,
    algorithm_name_from_cfg,
    annotate_diagnostic_frame,
    diagnostic_lines,
)


def _step_info() -> dict:
    return {
        "route_completion": 0.427,
        "ego_state": {"speed_km_h": 32.4, "yaw": 1.27},
        "rule_reward_vector": (0.0, -0.12, -0.063, 0.018),
        "rule_metadata": {
            "rule_names": [
                "collision_safety",
                "interaction_risk",
                "non_relaxable_compliance",
                "mission_progress",
            ]
        },
        "rule_components": {
            "collision": {"cost": 0.0},
            "rss": {"cost": 0.12},
            "ttc": {"cost": 0.063},
            "signal": {"cost": 0.0},
            "progress": {"cost": 0.018, "margin": 0.018},
        },
    }


def test_algorithm_name_reads_hydra_config() -> None:
    cfg = OmegaConf.create({"agent": {"planner": {"algorithm": {"name": "td3_sb3"}}}})
    assert algorithm_name_from_cfg(cfg) == "td3_sb3"


def test_diagnostic_lines_are_compact_and_show_subrules() -> None:
    state = DiagnosticState(algorithm="td3_sb3")
    state.update(-0.183)
    lines = diagnostic_lines(state=state, reward=-0.183, step_info=_step_info())

    assert lines[:4] == [
        "Algorithm: td3_sb3",
        "Step: 1   Speed: +32.4 km/h",
        "Heading: +1.27 rad   Route: 42.7%",
        "Reward: -0.18   Cumulative: -0.18",
    ]
    # RULEBOOK-V5.1 §3: the overlay labels the six levels L1..L6. The former
    # R1..R4 named v4.7's macro rules, which no longer exist.
    assert any("L2 m=-0.12" in line for line in lines)
    assert any("rss c=0.12" in line and "ttc c=0.06" in line for line in lines)
    assert not any("collision c=" in line for line in lines)
    assert not any("progress c=" in line for line in lines)


def test_diagnostic_lines_distinguish_not_applicable_from_satisfied() -> None:
    """TEST-RBCOST-022 / REQ-RBCOST-012.

    A NOT_APPLICABLE sub-rule (e.g. no signal control selected) and a
    SATISFIED one (control selected, currently green) both cost 0.00 and were
    indistinguishable in the overlay; the status must render differently.
    """
    info = _step_info()
    info["rule_components"]["signal"] = {"cost": 0.0, "status": "not_applicable"}
    state = DiagnosticState(algorithm="td3_sb3")
    not_applicable_lines = diagnostic_lines(state=state, reward=0.0, step_info=info)

    info["rule_components"]["signal"] = {
        "cost": 0.0,
        "status": "satisfied",
        "raw": {"pre_state": "GREEN", "post_state": "GREEN"},
        "diagnostics": {"post_delta_m": 12.3},
    }
    satisfied_lines = diagnostic_lines(state=state, reward=0.0, step_info=info)

    not_applicable_text = " ".join(not_applicable_lines)
    satisfied_text = " ".join(satisfied_lines)
    assert "signal c=0.00[n/a]" in not_applicable_text
    assert "signal c=0.00[n/a]" not in satisfied_text
    assert "GREEN" in satisfied_text
    assert "12.3m" in satisfied_text


def test_diagnostic_lines_tolerate_missing_optional_fields() -> None:
    state = DiagnosticState(algorithm="ppo_sb3")
    state.update(1.0)
    lines = diagnostic_lines(state=state, reward=1.0, step_info={})

    assert lines[1] == "Step: 1   Speed: - km/h"
    assert lines[2] == "Heading: - rad   Route: -"
    assert lines[3] == "Reward: +1.00   Cumulative: +1.00"


def test_diagnostic_lines_use_top_level_ego_fallbacks() -> None:
    state = DiagnosticState(algorithm="td3_sb3")
    state.update(0.0)
    lines = diagnostic_lines(
        state=state,
        reward=0.0,
        step_info={"speed": 18.5, "yaw": -0.4, "route_completion": 0.2},
    )
    assert lines[1] == "Step: 1   Speed: +18.5 km/h"
    assert lines[2] == "Heading: -0.40 rad   Route: 20.0%"


def test_annotator_preserves_frame_shape_and_rgb() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    state = DiagnosticState(algorithm="sac_sb3")
    state.update(-0.5)
    annotated = annotate_diagnostic_frame(
        frame,
        state=state,
        reward=-0.5,
        step_info=_step_info(),
        geometry={
            "route_past": [(4, 40), (20, 20)],
            "route_future": [(20, 20), (60, 4)],
            "target": (58, 5),
            "neighbors": [((30, 20), False), ((40, 20), True)],
        },
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated != frame)


def test_reward_state_accumulates_selected_reward_only() -> None:
    state = DiagnosticState(algorithm="td3_sb3")
    for reward in (0.5, -0.25, 0.75):
        state.update(reward)
    assert state.step == 3
    assert state.cumulative_reward == 1.0


def test_geometry_extraction_skips_vector_env_proxy_boundary() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry

    class VectorEnvSlotProxy:
        unwrapped = None

    proxy = VectorEnvSlotProxy()
    proxy.unwrapped = proxy
    assert diagnostic_geometry(proxy, {}) == {}


def test_geometry_extraction_uses_renderer_and_vehicle_fallbacks() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    class MissionSnap:
        s_m = 5.0

    class EnvSnap:
        mission_snapshot = MissionSnap()

    class Context:
        mission_route = RoutePolyline(((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)))
        snapshot = EnvSnap()

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = Context()

    geometry = diagnostic_geometry(Env(), {})

    assert "route_past" not in geometry
    assert len(geometry["route_future"]) >= 2


def test_geometry_extraction_uses_mission_gates_instead_of_route_checkpoints() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    class Runtime:
        gates = ()

    class MissionSnap:
        s_m = 5.0

    class EnvSnap:
        mission_snapshot = MissionSnap()

    class Context:
        mission_route = RoutePolyline(((0.0, 5.0, 0.0), (20.0, 5.0, 0.0)))
        snapshot = EnvSnap()

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = Context()
        _mission_runtime = Runtime()

    geometry = diagnostic_geometry(Env(), {})

    assert "checkpoints" not in geometry


def test_geometry_extraction_projects_mission_gates_with_progress_colours() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    from thesis_rl.mission.gates import GateGeometry
    from thesis_rl.mission.types import DirectedGate, LaneSpan, MissionSnapshot

    span = LaneSpan("a", 0.0, 10.0)

    class Runtime:
        gates = (
            DirectedGate(
                "gate:0",
                (span,),
                "a",
                1.0,
                GateGeometry(((3.0, 4.0), (3.0, 6.0)), (1.0, 0.0), 0.0),
            ),
            DirectedGate(
                "goal:final",
                (span,),
                "a",
                2.0,
                GateGeometry(((8.0, 4.0), (8.0, 6.0)), (1.0, 0.0), 0.0),
            ),
        )
        snapshot = MissionSnapshot("mission", 2, 1, 3.0, 0.5, True, False, False)

    class MissionSnap:
        s_m = 5.0

    class EnvSnap:
        mission_snapshot = MissionSnap()

    class Context:
        mission_route = RoutePolyline(((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)))
        snapshot = EnvSnap()

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = Context()
        _mission_runtime = Runtime()

    geometry = diagnostic_geometry(Env(), {})

    # Screen coordinates are camera-relative (`to_screen` subtracts the
    # camera's own screen offset, matching every other overlay in this
    # function); with the track agent at world (5, 5) and a 200x200 canvas,
    # world (3, 4) does not map to raw `pos2pix(3, 4)`.
    assert geometry["mission_gates"] == [
        {"line": [(80.0, 90.0), (80.0, 110.0)], "state": "passed", "final": False},
        {"line": [(130.0, 90.0), (130.0, 110.0)], "state": "pending", "final": True},
    ]


def test_geometry_extraction_renders_final_gate_segment_without_nested_geometry() -> None:
    """Regression: a route-coordinate mission's sole gate is a frozen
    ``FinalGateSegment`` (direct ``line_xy`` field), not a legacy
    ``DirectedGate`` (``line_xy`` nested under ``.geometry``). Without a
    fallback for the flat shape, ``getattr(gate, "geometry", None)`` is
    always ``None`` and no gate -- in particular the final gate -- is ever
    drawn for any PG or Waymo v1.1.1 mission.
    """
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    from thesis_rl.mission.types import FinalGateSegment, MissionSnapshot

    class Runtime:
        gates = (
            FinalGateSegment(
                ((8.0, 4.0), (8.0, 6.0)),
                (1.0, 0.0),
                0.0,
                "occurrence:0:a",
                "offline:pg:anchor_cross_section",
                "hash",
                "driving-mission-v1.1.1-anchor-builder",
            ),
        )
        snapshot = MissionSnapshot("mission", 2, 0, 3.0, 0.5, True, False, False)

    class MissionSnap:
        s_m = 5.0

    class EnvSnap:
        mission_snapshot = MissionSnap()

    class Context:
        mission_route = RoutePolyline(((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)))
        snapshot = EnvSnap()

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = Context()
        _mission_runtime = Runtime()

    geometry = diagnostic_geometry(Env(), {})

    assert geometry["mission_gates"] == [
        {"line": [(130.0, 90.0), (130.0, 110.0)], "state": "pending", "final": True},
    ]


def test_geometry_extraction_projects_ego_trail_when_given() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = None

    geometry = diagnostic_geometry(Env(), {}, ego_trail_world=[(0.0, 0.0), (2.0, 2.0), (5.0, 5.0)])

    assert len(geometry["ego_trail"]) == 3


def test_geometry_extraction_omits_ego_trail_when_not_given() -> None:
    from thesis_rl.runtime.io.video_diagnostics import diagnostic_geometry

    class Canvas:
        def pos2pix(self, x, y):
            return (int(x * 10), int(y * 10))

        def get_size(self):
            return (200, 200)

    class Vehicle:
        position = (5.0, 5.0, 0.0)

    class Renderer:
        _frame_canvas = Canvas()
        _screen_canvas = Canvas()
        position = None
        current_track_agent = Vehicle()
        target_agent_heading_up = False

    class Env:
        top_down_renderer = Renderer()
        causal_scene_context = None

    geometry = diagnostic_geometry(Env(), {})

    assert "ego_trail" not in geometry


def test_annotator_draws_mission_gates_and_ego_trail_without_crashing() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    state = DiagnosticState(algorithm="sac_sb3")
    state.update(-0.5)
    annotated = annotate_diagnostic_frame(
        frame,
        state=state,
        reward=-0.5,
        step_info=_step_info(),
        geometry={
            "mission_gates": [
                {"line": [(10, 10), (10, 30)], "state": "passed", "final": False},
                {"line": [(30, 10), (30, 30)], "state": "pending", "final": True},
            ],
            "ego_trail": [(5, 40), (15, 30), (25, 25)],
        },
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated != frame)


def test_route_polyline_from_lane_centerlines_exposes_lane_start_points() -> None:
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    route = RoutePolyline.from_lane_centerlines(
        (
            ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
            ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
            ((20.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
        )
    )
    assert route.lane_start_points_xyz == (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
    )


def test_route_polyline_plain_constructor_defaults_lane_start_points_empty() -> None:
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    assert route.lane_start_points_xyz == ()
