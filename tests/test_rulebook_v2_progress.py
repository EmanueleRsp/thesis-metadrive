import pytest

from thesis_rl.mission.types import MissionSnapshot
from thesis_rl.rulebook.v2.components.progress import (
    MISSION_PROGRESS_REFERENCE_SPEED_MPS,
    evaluate_progress,
)


def _snapshot(*, step: int, remaining_distance_m: float, mission_hash: str = "mission"):
    return MissionSnapshot(
        mission_hash=mission_hash,
        step_index=step,
        pending_gate_index=0,
        remaining_distance_m=remaining_distance_m,
        route_completion=0.0,
        reachable=True,
        mission_success=False,
        mission_unreachable=False,
    )


def test_progress_uses_signed_mission_distance_reduction_with_global_normalizer() -> None:
    result, delta, _ = evaluate_progress(
        pre_mission=_snapshot(step=4, remaining_distance_m=102.0),
        post_mission=_snapshot(step=5, remaining_distance_m=100.0),
        delta_t_s=0.1,
    )

    assert result.raw["mission_distance_delta_m"] == pytest.approx(2.0)
    assert result.cost == pytest.approx(2.0 / (MISSION_PROGRESS_REFERENCE_SPEED_MPS * 0.1))
    assert result.diagnostics["reference_speed_mps"] == MISSION_PROGRESS_REFERENCE_SPEED_MPS
    assert delta.writes == ()


def test_progress_is_cap_invariant_and_clips_signed_distance_reduction() -> None:
    forward, _, _ = evaluate_progress(
        pre_mission=_snapshot(step=0, remaining_distance_m=100.0),
        post_mission=_snapshot(step=1, remaining_distance_m=50.0),
        delta_t_s=0.1,
    )
    reverse, _, _ = evaluate_progress(
        pre_mission=_snapshot(step=1, remaining_distance_m=50.0),
        post_mission=_snapshot(step=2, remaining_distance_m=100.0),
        delta_t_s=0.1,
    )

    assert forward.cost == 1.0
    assert reverse.cost == -1.0


def test_progress_rejects_nonconsecutive_or_cross_mission_snapshots() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        evaluate_progress(
            pre_mission=_snapshot(step=0, remaining_distance_m=10.0),
            post_mission=_snapshot(step=2, remaining_distance_m=8.0),
            delta_t_s=0.1,
        )
    with pytest.raises(ValueError, match="identity"):
        evaluate_progress(
            pre_mission=_snapshot(step=0, remaining_distance_m=10.0, mission_hash="one"),
            post_mission=_snapshot(step=1, remaining_distance_m=8.0, mission_hash="two"),
            delta_t_s=0.1,
        )


def test_progress_accepts_terminal_unreachable_reset_noop() -> None:
    terminal = MissionSnapshot(
        mission_hash="mission",
        step_index=0,
        pending_gate_index=0,
        remaining_distance_m=0.0,
        route_completion=0.0,
        reachable=False,
        mission_success=False,
        mission_unreachable=True,
        reason="mission_unreachable",
    )

    result, delta, _ = evaluate_progress(
        pre_mission=terminal,
        post_mission=terminal,
        delta_t_s=0.1,
    )

    assert result.cost == 0.0
    assert result.raw["mission_distance_delta_m"] == 0.0
    assert result.diagnostics["terminal_mission_unreachable_noop"] is True
    assert delta.writes == ()
