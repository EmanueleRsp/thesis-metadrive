"""Small adapter around the ScenarioEnv/vehicle APIs used by the thesis env.

The adapter deliberately delegates to attributes exposed by the checked-out
MetaDrive version.  It does not introduce a second geometric road model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SceneContextAdapter:
    """Expose stable, defensive accessors for rulebook and termination code."""

    _ROAD_EDGE_BOUNDARY = "ROAD_EDGE_BOUNDARY"
    _ROAD_EDGE_SIDEWALK = "ROAD_EDGE_SIDEWALK"
    _GUARDRAIL = "GUARDRAIL"
    _FULL_FOOTPRINT_EPSILON_M2 = 1.0e-4

    def get_ego_vehicle(self, env: Any, vehicle_id: str | None = None) -> Any:
        agents = getattr(env, "agents", {})
        if vehicle_id is not None and vehicle_id in agents:
            return agents[vehicle_id]
        if not agents:
            return None
        return next(iter(agents.values()))

    def get_route_completion(self, env: Any, vehicle: Any) -> float | None:
        del env
        value = getattr(getattr(vehicle, "navigation", None), "route_completion", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def is_on_continuous_line(self, vehicle: Any) -> bool:
        return bool(
            getattr(vehicle, "on_yellow_continuous_line", False)
            or getattr(vehicle, "on_white_continuous_line", False)
        )

    def is_destination_reached(self, env: Any, vehicle: Any) -> bool:
        predicate = getattr(env, "_is_arrive_destination", None)
        if callable(predicate):
            return bool(predicate(vehicle))
        completion = self.get_route_completion(env, vehicle)
        return completion is not None and completion > 0.95

    def is_physically_out_of_road(self, env: Any, vehicle: Any) -> bool:
        """Use native surface and contact primitives, excluding route drift."""

        if bool(getattr(vehicle, "crash_sidewalk", False)):
            # MetaDrive's rectangular sidewalk probe reports ROAD_EDGE_BOUNDARY
            # through the generic ``crash_sidewalk`` flag. That boundary can be
            # touched while the vehicle is still on the drivable surface; only
            # an actual sidewalk/guardrail contact is a physical road exit.
            contacts = getattr(vehicle, "contact_results", None)
            if contacts is None:
                return True
            try:
                contacts = set(contacts)
            except TypeError:
                contacts = set()
            if contacts & {self._ROAD_EDGE_SIDEWALK, self._GUARDRAIL}:
                return True
            if self._ROAD_EDGE_BOUNDARY not in contacts:
                return True

        geometry_exit, _, _ = self._rulebook_full_footprint_exit(env)
        if geometry_exit:
            return True

        # ``navigation.current_lateral``, ``dist_to_*_side``, and ``on_lane``
        # all derive from ``current_ref_lanes`` in ScenarioNet's
        # TrajectoryNavigation. They describe deviation from the assigned
        # reference route, rather than physical contact with the road edge.
        # They remain diagnostics only: physical exit is established by native
        # sidewalk/guardrail contacts or by the canonical full-footprint
        # Rulebook geometry above.
        return False

    def _rulebook_full_footprint_exit(self, env: Any) -> tuple[bool, float | None, float | None]:
        """Return the Rulebook-backed full-footprint road-exit classification.

        ScenarioNet does not always emit a road-edge contact after the ego has
        left its mapped road.  The Rulebook adapter already owns the canonical
        live footprint and vertically compatible drivable surface, so reuse
        that exact geometry rather than inferring an exit from route-relative
        navigation fields.
        """

        adapter = getattr(env, "rulebook_v2_adapter", None)
        snapshotter = getattr(adapter, "snapshotter", None)
        cache = getattr(adapter, "initial_cache", None)
        if not callable(snapshotter) or cache is None:
            return False, None, None

        from thesis_rl.rulebook.v2.geometry.drivable import (
            DrivableLaneRecord,
            drivable_surface_for_ego,
        )

        ego = snapshotter(env).ego
        surface = drivable_surface_for_ego(
            ego_footprint=ego.footprint,
            ego_position_xy=ego.position_xy,
            ego_position_z=ego.position_z,
            lanes=tuple(
                DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
                for lane in cache.route_lanes
            ),
        )
        ego_area = float(ego.footprint.area)
        outside_area = float(ego.footprint.difference(surface).area)
        fully_outside = outside_area >= ego_area - self._FULL_FOOTPRINT_EPSILON_M2
        return fully_outside, outside_area, ego_area

    def get_physical_road_diagnostics(self, env: Any, vehicle: Any) -> dict[str, Any]:
        """Return native values needed to audit physical-road termination."""

        navigation = getattr(vehicle, "navigation", None)
        contacts = getattr(vehicle, "contact_results", ()) or ()
        geometry_exit, outside_area, ego_area = self._rulebook_full_footprint_exit(env)
        return {
            "route_lateral": getattr(navigation, "current_lateral", None),
            "dist_to_left_side": getattr(vehicle, "dist_to_left_side", None),
            "dist_to_right_side": getattr(vehicle, "dist_to_right_side", None),
            "on_lane": getattr(vehicle, "on_lane", None),
            "contact_results": sorted(str(value) for value in contacts),
            "geometric_full_footprint_exit": geometry_exit,
            "geometric_outside_area_m2": outside_area,
            "geometric_ego_area_m2": ego_area,
        }

    def get_native_out_of_road(self, env: Any, vehicle: Any) -> bool:
        predicate = getattr(env, "_is_out_of_road", None)
        return bool(predicate(vehicle)) if callable(predicate) else False

    def get_termination_reason(
        self,
        env: Any,
        vehicle: Any,
        done_info: Mapping[str, Any] | None = None,
    ) -> str | None:
        del env, vehicle
        if done_info is None:
            return None
        reasons = (
            ("arrive_dest", "success"),
            ("crash_human", "crash_human"),
            ("crash_vehicle", "crash_vehicle"),
            ("crash_object", "crash_object"),
            ("crash_building", "crash_building"),
            ("crash_sidewalk", "crash_sidewalk"),
            ("crash", "collision"),
            ("out_of_road", "out_of_road"),
            ("max_step", "time_limit"),
        )
        for key, reason in reasons:
            if bool(done_info.get(key, False)):
                return reason
        return None

    def get_traffic_controls(self, env: Any) -> tuple[Any, ...]:
        manager = getattr(getattr(env, "engine", None), "light_manager", None)
        if manager is None:
            return ()
        objects = getattr(manager, "traffic_lights", None)
        if isinstance(objects, Mapping):
            return tuple(objects.values())
        if objects is None:
            return ()
        try:
            return tuple(objects)
        except TypeError:
            return ()

    def get_nearby_agents(self, env: Any, vehicle: Any) -> tuple[Any, ...]:
        del vehicle
        agents = getattr(env, "agents", {})
        return tuple(agents.values()) if isinstance(agents, Mapping) else ()

    def get_ego_dimensions(self, vehicle: Any) -> tuple[float, float] | None:
        try:
            length = float(getattr(vehicle, "LENGTH"))
            width = float(getattr(vehicle, "WIDTH"))
        except (AttributeError, TypeError, ValueError):
            return None
        return length, width
