"""Shared diagnostic overlays for live and replayed evaluation frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from thesis_rl.runtime.io.video_utils import to_uint8_rgb


_MACRO_ALIASES = {
    "collision_impact": "R1",
    "dynamic_interaction_safety": "R2",
    "road_traffic_compliance": "R3",
    "route_progress": "R4",
}
_SUBRULE_ALIASES = {
    "collision_impact": "collision",
    "dynamic_interaction_safety": "dynamic",
    "road_traffic_compliance": "road",
    "route_progress": "progress",
    "clearance": "CLR",
    "vehicle_yield": "yield",
    "wrong_way": "wrongway",
}


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else default
    return default


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:+.{digits}f}"


def _fmt_percent(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{100.0 * number:.1f}%"


def _get(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def algorithm_name_from_cfg(cfg: Any) -> str:
    """Return the configured algorithm name without coupling to Hydra types."""

    current = cfg
    for key in ("agent", "planner", "algorithm"):
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    if isinstance(current, Mapping):
        value = current.get("name")
    else:
        value = getattr(current, "name", None)
    return str(value or "unknown")


@dataclass(slots=True)
class DiagnosticState:
    """Episode-local state required to annotate one frame at a time."""

    algorithm: str = "unknown"
    cumulative_reward: float = 0.0
    step: int = 0

    def update(self, reward: float) -> None:
        self.step += 1
        self.cumulative_reward += float(reward)


def _rule_lines(step_info: Mapping[str, Any]) -> list[str]:
    vector = step_info.get("rule_reward_vector")
    metadata = step_info.get("rule_metadata")
    names = metadata.get("rule_names") if isinstance(metadata, Mapping) else None
    lines: list[str] = []
    if isinstance(vector, (list, tuple, np.ndarray)) and isinstance(names, list):
        for name, value in zip(names, vector):
            macro = str(name)
            prefix = _MACRO_ALIASES.get(macro, macro[:8])
            lines.append(f"{prefix} {_SUBRULE_ALIASES.get(macro, macro[:10])} {_fmt(value)}")

    components = step_info.get("rule_components")
    if not isinstance(components, Mapping):
        return lines
    subrules: list[str] = []
    for name, payload in components.items():
        if not isinstance(payload, Mapping):
            continue
        cost = _get(payload, "cost", "margin", "value")
        if cost is None:
            continue
        label = _SUBRULE_ALIASES.get(str(name), str(name)[:9])
        subrules.append(f"{label} {_fmt(cost)}")
    if subrules:
        # Keep all subrules, but put them on compact continuation rows.
        for start in range(0, len(subrules), 4):
            lines.append("  " + " | ".join(subrules[start : start + 4]))
    return lines


def diagnostic_lines(
    *,
    state: DiagnosticState,
    reward: float,
    step_info: Any,
) -> list[str]:
    """Build deterministic panel lines from the selected transition reward."""

    info = step_info if isinstance(step_info, Mapping) else {}
    ego = info.get("ego_state") if isinstance(info.get("ego_state"), Mapping) else {}
    speed = _get(ego, "speed_km_h", "velocity_km_h")
    if speed is None:
        speed_mps = _get(ego, "speed_m_s", "speed")
        speed = None if speed_mps is None else 3.6 * float(speed_mps)
    heading = _get(ego, "yaw", "heading", "heading_rad")
    route = _get(info, "route_completion", "route_completion_ratio", "progress")

    lines = [
        f"Algorithm: {state.algorithm}",
        f"Step: {state.step}   Speed: {_fmt(speed, 1)} km/h",
        f"Heading: {_fmt(heading, 2)} rad   Route: {_fmt_percent(route)}",
        f"Reward: {_fmt(reward)}   Cumulative: {_fmt(state.cumulative_reward)}",
    ]
    lines.extend(_rule_lines(info))
    return lines


def _point_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        x, y = _number(value[0]), _number(value[1])
        if x is not None and y is not None:
            return x, y
    return None


def _find_actor_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        worst = value.get("worst_actor_id")
        if worst is not None:
            found.add(str(worst))
        for child in value.values():
            found.update(_find_actor_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_actor_ids(child))
    return found


def diagnostic_geometry(env: Any, step_info: Any) -> dict[str, Any]:
    """Extract exact optional world-to-screen primitives from a live renderer.

    The helper deliberately returns an empty payload for unsupported camera
    modes or environments. It never reconstructs geometry from observations.
    """

    info = step_info if isinstance(step_info, Mapping) else {}
    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "top_down_renderer", None)
    canvas = getattr(renderer, "_frame_canvas", None)
    screen = getattr(renderer, "_screen_canvas", None)
    track_agent = getattr(renderer, "current_track_agent", None)
    pos2pix = getattr(canvas, "pos2pix", None)
    if renderer is None or canvas is None or screen is None or not callable(pos2pix):
        return {}
    if bool(getattr(renderer, "target_agent_heading_up", False)):
        return {}
    screen_width, screen_height = screen.get_size()
    camera_position = getattr(renderer, "position", None) or getattr(track_agent, "position", None)
    if camera_position is None:
        return {}
    camera_pixel = pos2pix(float(camera_position[0]), float(camera_position[1]))
    offset = (camera_pixel[0] - screen_width / 2.0, camera_pixel[1] - screen_height / 2.0)

    def to_screen(point: Any) -> tuple[float, float] | None:
        xy = _point_xy(point)
        if xy is None:
            return None
        world_pixel = pos2pix(*xy)
        return float(world_pixel[0] - offset[0]), float(world_pixel[1] - offset[1])

    geometry: dict[str, Any] = {}
    context = getattr(base, "causal_scene_context", None)
    route = getattr(context, "route_polyline", None)
    if route is None:
        adapter = getattr(base, "rulebook_v2_adapter", None)
        cache = getattr(adapter, "initial_cache", None)
        route = getattr(cache, "route_polyline", None)
    route_points = getattr(route, "points_xyz", None)
    ego = info.get("ego_state") if isinstance(info.get("ego_state"), Mapping) else {}
    ego_position = _point_xy(_get(ego, "position"))
    if route is not None and route_points and ego_position is not None:
        try:
            projection = route.project(ego_position)
            past_world = list(route_points[: projection.segment_index + 1])
            past_world.append(route.point_at(projection.s_m))
            future_world = [
                route.point_at(projection.s_m),
                *route_points[projection.segment_index + 1 :],
            ]
            geometry["route_past"] = [
                point for point in (to_screen(p) for p in past_world) if point
            ]
            geometry["route_future"] = [
                point for point in (to_screen(p) for p in future_world) if point
            ]
        except Exception:
            pass

    target = to_screen(info.get("target_point"))
    if target is not None:
        geometry["target"] = target

    neighbors = info.get("neighbors")
    if isinstance(neighbors, (list, tuple)):
        critical_ids = _find_actor_ids(info.get("rule_components"))
        critical_ids.update(_find_actor_ids(info.get("rulebook")))
        geometry["neighbors"] = [
            (point, str(neighbor.get("entity_id")) in critical_ids)
            for neighbor in neighbors
            if isinstance(neighbor, Mapping)
            for point in [to_screen(neighbor.get("position"))]
            if point is not None
        ]
    return geometry


def annotate_diagnostic_frame(
    frame: np.ndarray,
    *,
    state: DiagnosticState,
    reward: float,
    step_info: Any,
    geometry: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Draw the shared semi-transparent panel without changing frame size."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # pragma: no cover - Pillow is an existing video dependency
        return to_uint8_rgb(np.asarray(frame))

    image = Image.fromarray(to_uint8_rgb(np.asarray(frame))).convert("RGBA")
    lines = diagnostic_lines(state=state, reward=reward, step_info=step_info)
    font = ImageFont.load_default()
    line_height = 13
    padding = 7
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_width = max(
        (measure.textlength(line, font=font) for line in lines),
        default=200,
    )
    available_width = max(image.width - 12, 1)
    box_width = min(available_width, max(250, int(max_width) + 2 * padding))
    box_height = padding * 2 + line_height * len(lines)
    x0 = max(6, image.width - box_width - 6)
    y0 = max(6, image.height - box_height - 6)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    geometry = geometry or {}

    def line(points: Any, color: tuple[int, int, int, int], width: int = 2) -> None:
        if isinstance(points, (list, tuple)) and len(points) >= 2:
            draw.line([tuple(map(float, point)) for point in points], fill=color, width=width)

    line(geometry.get("route_future"), (50, 130, 255, 125), width=2)
    line(geometry.get("route_past"), (50, 190, 80, 190), width=3)
    target = geometry.get("target")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        x, y = float(target[0]), float(target[1])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(30, 80, 220, 210), width=2)
    for point, critical in geometry.get("neighbors", ()):
        x, y = float(point[0]), float(point[1])
        color = (220, 40, 35, 225) if critical else (240, 145, 20, 210)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
    draw.rounded_rectangle(
        (x0, y0, x0 + box_width, y0 + box_height),
        radius=4,
        fill=(255, 255, 255, 205),
        outline=(255, 255, 255, 235),
        width=1,
    )
    y = y0 + padding
    for line in lines:
        draw.text((x0 + padding, y), line, fill=(0, 0, 0, 255), font=font)
        y += line_height
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))
