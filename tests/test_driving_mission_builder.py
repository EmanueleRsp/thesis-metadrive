from __future__ import annotations

from thesis_rl.mission.builder import NormalizedLane, build_driving_mission


def _lanes() -> dict[str, NormalizedLane]:
    return {
        "a": NormalizedLane("a", 10.0, ("b",)),
        "b": NormalizedLane("b", 10.0, ("c", "d"), ("parallel_b",)),
        "parallel_b": NormalizedLane("parallel_b", 10.0, ("c",)),
        "c": NormalizedLane("c", 10.0),
        "d": NormalizedLane("d", 10.0),
    }


def test_builder_consolidates_unambiguous_chain_and_places_gate_at_branch() -> None:
    record = build_driving_mission("uid", ("a", "b", "c"), _lanes(), final_goal_lane_id="c", final_goal_s_m=6.0)

    assert len(record.sections) == 1
    assert record.sections[0].preferred_span.lane_id == "a"
    assert record.sections[0].exit_gate.lane_id == "b"
    assert record.final_goal.lane_id == "c"
    assert record.final_goal.s_m == 6.0


def test_builder_allows_only_lateral_lane_that_reaches_next_gate() -> None:
    record = build_driving_mission("uid", ("a", "b", "c"), _lanes(), final_goal_lane_id="c", final_goal_s_m=6.0)

    assert {span.lane_id for span in record.sections[0].allowed_spans} == {"a", "b", "parallel_b"}
