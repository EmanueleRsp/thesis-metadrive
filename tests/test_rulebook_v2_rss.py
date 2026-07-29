from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.context.live_adapter import actor_snapshot_from_payload
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.transition import _rss_candidates
from thesis_rl.rulebook.v2.types import ActorClass

import pytest

from thesis_rl.rulebook.v2.components.rss import (
    RSSCalibrationArtifact,
    RSSCandidate,
    evaluate_rss,
    safe_distance_m,
)
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError


def test_rss_safe_distance_and_continuous_deficit() -> None:
    calibration = RSSCalibrationArtifact("ego-hash", 3.0)
    safe = safe_distance_m(ego_speed_mps=5.0, front_speed_mps=5.0, ego_brake_mps2=3.0)
    result, _, _ = evaluate_rss(
        scenario_id="scenario", step_index=1,
        candidates=(RSSCandidate("front", safe, 5.0, 5.0),),
        calibration=calibration, expected_config_hash="ego-hash",
    )
    assert result.cost == pytest.approx(0.0)
    deficient, _, _ = evaluate_rss(
        scenario_id="scenario", step_index=1,
        candidates=(RSSCandidate("front", safe / 2.0, 5.0, 5.0),),
        calibration=calibration, expected_config_hash="ego-hash",
    )
    assert deficient.cost == pytest.approx(0.5)


def test_rss_missing_or_mismatched_calibration_fails_fast() -> None:
    candidate = (RSSCandidate("front", 1.0, 5.0, 5.0),)
    with pytest.raises(RulebookEvaluationError, match="missing"):
        evaluate_rss(
            scenario_id="scenario", step_index=1, candidates=candidate,
            calibration=None, expected_config_hash="ego-hash",
        )
    with pytest.raises(RulebookEvaluationError, match="hash"):
        evaluate_rss(
            scenario_id="scenario", step_index=1, candidates=candidate,
            calibration=RSSCalibrationArtifact("other", 3.0), expected_config_hash="ego-hash",
        )


def test_rss_longitudinal_keeps_a_lead_vehicle_on_the_successor_lane() -> None:
    """TEST-EF-18 / REQ-EF-11 (v4.7 6.2.1, 2.9.3).

    Exact ``lane_id`` equality dropped a lead vehicle the moment it crossed a
    lane-segment boundary of the very same road, which on a PG route happens
    every few seconds because the map is segmented per block.
    """

    lane_a = RouteLaneRecord(
        "A",
        Polygon([(-10, -1.75), (50, -1.75), (50, 1.75), (-10, 1.75)]),
        RoutePolyline(tuple((float(x), 0.0, 0.0) for x in range(-10, 51, 2))),
        ("B",),
    )
    lane_b = RouteLaneRecord(
        "B",
        Polygon([(50, -1.75), (120, -1.75), (120, 1.75), (50, 1.75)]),
        RoutePolyline(tuple((float(x), 0.0, 0.0) for x in range(50, 121, 2))),
    )
    route = RoutePolyline.from_lane_centerlines(
        (lane_a.centerline.points_xyz, lane_b.centerline.points_xyz)
    )
    lanes = (lane_a, lane_b)

    def vehicle(actor_id, x):
        return actor_snapshot_from_payload(
            {
                "actor_id": actor_id,
                "actor_class": ActorClass.VEHICLE,
                "position_xy": (x, 0.0),
                "position_z": 0.0,
                "heading_rad": 0.0,
                "velocity_xy": (20.0, 0.0),
                "length_m": 4.515,
                "width_m": 1.852,
                "live_lane_id": None,
                "configured_speed_cap_mps": 30.0,
            }
        )

    # Ego near the end of lane A, lead just past the boundary into lane B.
    candidates = _rss_candidates(
        ego=vehicle("ego", 45.0),
        actors=(vehicle("lead", 60.0),),
        route=route,
        route_lanes=lanes,
    )
    assert [candidate.actor_id for candidate in candidates] == ["lead"]
    assert candidates[0].gap_m == pytest.approx(15.0 - 4.515, abs=1.0e-6)
