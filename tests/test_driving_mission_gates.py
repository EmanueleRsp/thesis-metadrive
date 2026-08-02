from __future__ import annotations

from thesis_rl.mission.gates import GateGeometry, directed_gate_crossed
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box


def test_directed_gate_requires_forward_swept_front_bumper_crossing() -> None:
    gate = GateGeometry(((0.0, -3.0), (0.0, 3.0)), (1.0, 0.0), 0.0)
    pre = oriented_bounding_box(center_xy=(-2.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    post = oriented_bounding_box(center_xy=(1.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    assert directed_gate_crossed(
        gate,
        pre_footprint=pre,
        post_footprint=post,
        pre_heading_rad=0.0,
        post_heading_rad=0.0,
        post_ego_z_m=0.0,
    )


def test_directed_gate_rejects_reverse_or_wrong_level_crossing() -> None:
    gate = GateGeometry(((0.0, -3.0), (0.0, 3.0)), (1.0, 0.0), 0.0)
    pre = oriented_bounding_box(
        center_xy=(1.0, 0.0), heading_rad=3.141592653589793, length_m=4.0, width_m=2.0
    )
    post = oriented_bounding_box(
        center_xy=(-2.0, 0.0), heading_rad=3.141592653589793, length_m=4.0, width_m=2.0
    )
    assert not directed_gate_crossed(
        gate,
        pre_footprint=pre,
        post_footprint=post,
        pre_heading_rad=3.141592653589793,
        post_heading_rad=3.141592653589793,
        post_ego_z_m=0.0,
    )
    assert not directed_gate_crossed(
        gate,
        pre_footprint=pre,
        post_footprint=post,
        pre_heading_rad=3.141592653589793,
        post_heading_rad=3.141592653589793,
        post_ego_z_m=4.0,
    )
