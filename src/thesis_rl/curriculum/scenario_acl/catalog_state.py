"""Per-arm coverage-cycle state over the frozen ScenarioNet catalog.

ACL `v1.3` REQ-009 (`DEC-009`, `ADR-032`): the training catalog is finite and
curated, so Generate draws sample *without replacement within a coverage cycle*
instead of i.i.d. with replacement. Each arm `i` owns a visited set `V_i` and a
cycle counter `c_i`; when the arm's admissible candidate set is exhausted the
cycle closes (`c_i <- c_i + 1`, `V_i <- {}`) and sampling restarts over the full
pool.

This state is a *view* over the catalog: it stores scenario UIDs only, never
scenario content, and it never mutates the frozen catalog or the arm partition.
It is run-local and is not comparable across runs (`ADR-032` consequences).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

COVERAGE_STATE_SCHEMA = "acl_coverage_v1"


class ScenarioCatalogVisitState:
    """Per-arm visited sets and coverage-cycle counters (REQ-009).

    ``mark_visited`` is applied at *selection* time rather than at commit time,
    so a record that is in flight cannot be drawn a second time inside the same
    cycle (`ADR-016` vectorized execution). A record whose episode ends in a
    typed data-abort (`ADR-024`) therefore stays marked as visited, which is the
    intended behaviour: a quarantined record must not be re-drawn.
    """

    def __init__(self, arm_names: Sequence[str]) -> None:
        names = [str(name) for name in arm_names]
        if not names:
            raise ValueError("ScenarioCatalogVisitState requires at least one arm name.")
        if len(set(names)) != len(names):
            raise ValueError("ScenarioCatalogVisitState arm names must be unique.")
        self._arm_names: tuple[str, ...] = tuple(names)
        self._visited: dict[str, set[str]] = {name: set() for name in self._arm_names}
        self._cycle_id: dict[str, int] = {name: 0 for name in self._arm_names}
        # Diagnostics only: never consulted by selection (`RAT-010` keeps the
        # intra-arm draw uniform, so no per-record priority may exist).
        self._num_generate_visits: dict[str, int] = {}
        self._last_generate_episode_id: dict[str, int] = {}

    @property
    def arm_names(self) -> tuple[str, ...]:
        return self._arm_names

    def _require_arm(self, arm_name: str) -> str:
        name = str(arm_name)
        if name not in self._visited:
            raise KeyError(f"Unknown ACL arm for coverage state: {name!r}")
        return name

    def is_visited(self, scenario_uid: str, *, arm_name: str) -> bool:
        """Return whether ``scenario_uid`` is already visited in the arm's current cycle."""

        return str(scenario_uid) in self._visited[self._require_arm(arm_name)]

    def visited_uids(self, arm_name: str) -> frozenset[str]:
        return frozenset(self._visited[self._require_arm(arm_name)])

    def visited_count(self, arm_name: str) -> int:
        return len(self._visited[self._require_arm(arm_name)])

    def cycle_id(self, arm_name: str) -> int:
        return int(self._cycle_id[self._require_arm(arm_name)])

    def mark_visited(self, scenario_uid: str, *, arm_name: str, episode_id: int) -> None:
        """Record a Generate selection of ``scenario_uid`` within the arm's cycle."""

        name = self._require_arm(arm_name)
        uid = str(scenario_uid)
        self._visited[name].add(uid)
        self._num_generate_visits[uid] = self._num_generate_visits.get(uid, 0) + 1
        self._last_generate_episode_id[uid] = int(episode_id)

    def close_cycle(self, arm_name: str) -> int:
        """Close the arm's coverage cycle and return the new cycle id.

        Only the given arm is affected: coverage cycles are per arm and no
        global cycle exists (REQ-009).
        """

        name = self._require_arm(arm_name)
        self._cycle_id[name] = int(self._cycle_id[name]) + 1
        self._visited[name] = set()
        return int(self._cycle_id[name])

    def num_generate_visits(self, scenario_uid: str) -> int:
        """Diagnostics: lifetime Generate selections of a record across all cycles."""

        return int(self._num_generate_visits.get(str(scenario_uid), 0))

    def last_generate_episode_id(self, scenario_uid: str) -> int | None:
        """Diagnostics: episode id of the most recent Generate selection, if any."""

        value = self._last_generate_episode_id.get(str(scenario_uid))
        return None if value is None else int(value)

    def coverage_summary(self) -> dict[str, dict[str, int]]:
        """Per-arm cycle id and visited count, for chunk-boundary diagnostics."""

        return {
            name: {
                "cycle_id": int(self._cycle_id[name]),
                "visited_in_cycle": len(self._visited[name]),
            }
            for name in self._arm_names
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": COVERAGE_STATE_SCHEMA,
            "arm_names": list(self._arm_names),
            "visited": {name: sorted(self._visited[name]) for name in self._arm_names},
            "cycle_id": {name: int(self._cycle_id[name]) for name in self._arm_names},
            "num_generate_visits": {
                uid: int(count) for uid, count in sorted(self._num_generate_visits.items())
            },
            "last_generate_episode_id": {
                uid: int(episode_id)
                for uid, episode_id in sorted(self._last_generate_episode_id.items())
            },
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "ScenarioCatalogVisitState":
        schema = payload.get("schema")
        if schema != COVERAGE_STATE_SCHEMA:
            raise ValueError(
                "Incompatible Scenario ACL coverage state: expected schema "
                f"{COVERAGE_STATE_SCHEMA!r}, got {schema!r}. Coverage cycles were introduced "
                "by ADR-032/`DEC-009`; earlier runs have no compatible state and require a "
                "fresh run."
            )
        arm_names = payload.get("arm_names")
        if not isinstance(arm_names, Iterable) or isinstance(arm_names, (str, bytes)):
            raise TypeError("Scenario ACL coverage state 'arm_names' must be a sequence.")
        state = cls([str(name) for name in arm_names])
        visited = payload.get("visited", {})
        if not isinstance(visited, dict):
            raise TypeError("Scenario ACL coverage state 'visited' must be a mapping.")
        for name, uids in visited.items():
            key = state._require_arm(str(name))
            if not isinstance(uids, Iterable) or isinstance(uids, (str, bytes)):
                raise TypeError("Scenario ACL coverage state visited entries must be sequences.")
            state._visited[key] = {str(uid) for uid in uids}
        cycle_id = payload.get("cycle_id", {})
        if not isinstance(cycle_id, dict):
            raise TypeError("Scenario ACL coverage state 'cycle_id' must be a mapping.")
        for name, value in cycle_id.items():
            key = state._require_arm(str(name))
            if int(value) < 0:
                raise ValueError("Scenario ACL coverage cycle ids must be non-negative.")
            state._cycle_id[key] = int(value)
        visits = payload.get("num_generate_visits", {})
        if not isinstance(visits, dict):
            raise TypeError("Scenario ACL coverage state 'num_generate_visits' must be a mapping.")
        state._num_generate_visits = {str(uid): int(count) for uid, count in visits.items()}
        last_seen = payload.get("last_generate_episode_id", {})
        if not isinstance(last_seen, dict):
            raise TypeError(
                "Scenario ACL coverage state 'last_generate_episode_id' must be a mapping."
            )
        state._last_generate_episode_id = {
            str(uid): int(episode_id) for uid, episode_id in last_seen.items()
        }
        return state
