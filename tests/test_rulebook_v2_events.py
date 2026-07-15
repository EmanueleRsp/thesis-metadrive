from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.events import detect_zone_transition


def test_zone_event_uses_pre_state_and_swept_front_bumper() -> None:
    zone = Polygon(((2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0)))
    pre = Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)))
    post = Polygon(((3.0, -0.5), (4.0, -0.5), (4.0, 0.5), (3.0, 0.5)))
    swept = Polygon(((1.0, -0.5), (3.0, -0.5), (3.0, 0.5), (1.0, 0.5)))
    events = detect_zone_transition(
        pre_ego_footprint=pre,
        post_ego_footprint=post,
        swept_front_bumper=swept,
        zone=zone,
        pre_front_s_m=1.0,
        post_front_s_m=3.0,
        zone_entry_s_m=2.0,
    )
    assert events.entered
    assert not events.pre_occupied
    assert events.post_occupied
