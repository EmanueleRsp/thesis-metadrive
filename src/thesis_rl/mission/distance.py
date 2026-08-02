"""Deterministic non-negative lane-graph distance queries."""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class LaneGraph:
    lengths_m: Mapping[str, float]
    successors: Mapping[str, tuple[str, ...]]
    lateral_successors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lengths_m or any(
            not lane or length <= 0.0 for lane, length in self.lengths_m.items()
        ):
            raise ValueError("lane graph requires positive lane lengths")
        if any(
            successor not in self.lengths_m
            for values in (*self.successors.values(), *self.lateral_successors.values())
            for successor in values
        ):
            raise ValueError("lane graph successor is unknown")

    def distance(
        self, lane_id: str, s_m: float, target_lane_id: str, target_s_m: float
    ) -> float | None:
        result = self.shortest_path(lane_id, s_m, target_lane_id, target_s_m)
        return None if result is None else result[0]

    def shortest_path(
        self, lane_id: str, s_m: float, target_lane_id: str, target_s_m: float
    ) -> tuple[float, tuple[str, ...]] | None:
        """Return metric distance and a lexicographically stable legal recovery path."""
        if (
            lane_id not in self.lengths_m
            or target_lane_id not in self.lengths_m
            or not 0.0 <= s_m <= self.lengths_m[lane_id]
            or not 0.0 <= target_s_m <= self.lengths_m[target_lane_id]
        ):
            return None
        if lane_id == target_lane_id and s_m <= target_s_m:
            return target_s_m - s_m, (lane_id,)
        initial_cost = self.lengths_m[lane_id] - s_m
        queue: list[tuple[float, tuple[str, ...], str]] = [(initial_cost, (lane_id,), lane_id)]
        best: dict[str, tuple[float, tuple[str, ...]]] = {lane_id: (initial_cost, (lane_id,))}
        while queue:
            cost, path, current = heapq.heappop(queue)
            if (cost, path) != best[current]:
                continue
            options = tuple(self.successors.get(current, ())) + tuple(
                self.lateral_successors.get(current, ())
            )
            for successor in sorted(set(options)):
                candidate_path = (*path, successor)
                if successor == target_lane_id:
                    return cost + target_s_m, candidate_path
                candidate = cost + self.lengths_m[successor]
                previous = best.get(successor)
                if previous is None or (candidate, candidate_path) < previous:
                    best[successor] = candidate, candidate_path
                    heapq.heappush(queue, (candidate, candidate_path, successor))
        return None
