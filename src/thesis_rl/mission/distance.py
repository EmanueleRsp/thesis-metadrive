"""Deterministic non-negative lane-graph distance queries."""

from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class LaneGraph:
    lengths_m: Mapping[str, float]
    successors: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.lengths_m or any(
            not lane or length <= 0.0 for lane, length in self.lengths_m.items()
        ):
            raise ValueError("lane graph requires positive lane lengths")
        if any(
            successor not in self.lengths_m
            for values in self.successors.values()
            for successor in values
        ):
            raise ValueError("lane graph successor is unknown")

    def distance(
        self, lane_id: str, s_m: float, target_lane_id: str, target_s_m: float
    ) -> float | None:
        if (
            lane_id not in self.lengths_m
            or target_lane_id not in self.lengths_m
            or not 0.0 <= s_m <= self.lengths_m[lane_id]
            or not 0.0 <= target_s_m <= self.lengths_m[target_lane_id]
        ):
            return None
        if lane_id == target_lane_id and s_m <= target_s_m:
            return target_s_m - s_m
        queue: list[tuple[float, str]] = [(self.lengths_m[lane_id] - s_m, lane_id)]
        best = {lane_id: self.lengths_m[lane_id] - s_m}
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != best[current]:
                continue
            for successor in sorted(self.successors.get(current, ())):
                if successor == target_lane_id:
                    return cost + target_s_m
                candidate = cost + self.lengths_m[successor]
                if candidate < best.get(successor, float("inf")):
                    best[successor] = candidate
                    heapq.heappush(queue, (candidate, successor))
        return None
