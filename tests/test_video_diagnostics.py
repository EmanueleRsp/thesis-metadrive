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
                "collision_impact",
                "dynamic_interaction_safety",
                "road_traffic_compliance",
                "route_progress",
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
    assert any("R2 m=-0.12" in line for line in lines)
    assert any("rss c=0.12" in line and "ttc c=0.06" in line for line in lines)
    assert not any("collision c=" in line for line in lines)
    assert not any("progress c=" in line for line in lines)


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
