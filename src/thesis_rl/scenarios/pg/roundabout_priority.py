"""Derive vehicle-yield roundabout priority records from a live PG map.

Populates the ``rulebook_vehicle_yield.roundabout_priorities`` scenario
metadata schema consumed by
``thesis_rl.rulebook.v2.context.static_adapter.vehicle_yield_records_from_metadata``,
from MetaDrive's own authoritative ``Roundabout`` block structure (never
inferred from generic topology), so that ``vehicle_yield`` can grant
circulating traffic priority over entering traffic in PG-generated
scenarios.
"""

from __future__ import annotations

import re
from typing import Any


def roundabout_priority_records(map_obj: Any) -> list[dict[str, str]]:
    """Return one record per (entry lane, circulating lane) pair.

    For every ``Roundabout`` block instance (``block.ID == "O"``) in
    ``map_obj.blocks``, pairs every lane that merges into the closed
    circulating ring with every lane belonging to that ring.

    Node naming is MetaDrive's own authoritative PG structure
    (``PGBlock.node(block_idx, part_idx, road_idx)`` ==
    ``f"{block_idx}{ID}{part_idx}_{road_idx}_"``): road_idx 0/1 nodes form
    the closed ring; road_idx 2/3 nodes are the socket branches. An "entry"
    edge is any edge landing on a road_idx-0 ring node whose origin node is
    not itself part of this block's own ring (either an external
    predecessor block, or one of the block's own socket entry roads).
    """

    records: list[dict[str, str]] = []
    for block in map_obj.blocks:
        if block.ID != "O":
            continue
        own_node = re.compile(rf"^-?{re.escape(block.name)}(\d+)_(\d+)_$")

        def _road_idx(node: str) -> int | None:
            match = own_node.match(node)
            return None if match is None else int(match.group(2))

        ring_lane_ids: list[str] = []
        entry_lane_ids: list[str] = []
        for from_node, to_dict in block.block_network.graph.items():
            for to_node, lanes in to_dict.items():
                from_idx, to_idx = _road_idx(from_node), _road_idx(to_node)
                is_ring = from_idx in (0, 1) and to_idx in (0, 1)
                is_entry = to_idx == 0 and from_idx not in (0, 1)
                if not (is_ring or is_entry):
                    continue
                for lane in lanes:
                    lane_id = str(lane.index)
                    (ring_lane_ids if is_ring else entry_lane_ids).append(lane_id)
        for entry_lane_id in entry_lane_ids:
            for circulating_lane_id in ring_lane_ids:
                records.append(
                    {
                        "component_id": block.name,
                        "entry_lane_id": entry_lane_id,
                        "circulating_lane_id": circulating_lane_id,
                    }
                )
    return records
