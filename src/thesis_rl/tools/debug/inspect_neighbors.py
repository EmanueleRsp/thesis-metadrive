#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

from thesis_rl.common.paths import default_output_path_str
from thesis_rl.runtime.wiring.builders import build_env


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "conf" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing conf/config.yaml. "
        f"Start path: {start}"
    )


def _xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    return arr[:2]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _jsonable(obj: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return "<MAX_DEPTH>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x, depth + 1, max_depth) for x in obj[:20]]
    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:80]:
            out[str(k)] = _jsonable(v, depth + 1, max_depth)
        return out
    # shapely o oggetti custom
    if hasattr(obj, "geom_type"):
        try:
            return {"geom_type": str(obj.geom_type), "bounds": list(obj.bounds)}
        except Exception:
            return {"geom_type": str(getattr(obj, "geom_type", "unknown"))}
    if hasattr(obj, "__dict__"):
        out = {"<type>": type(obj).__name__, "<attrs>": {}}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
                if callable(val):
                    continue
                out["<attrs>"][name] = _jsonable(val, depth + 1, max_depth)
            except Exception:
                continue
        return out
    return f"<{type(obj).__name__}>"


def _neighbor_diag(ego_state: dict[str, Any], neighbor: dict[str, Any]) -> dict[str, Any]:
    ego_pos = _xy(ego_state.get("position"))
    n_pos = _xy(neighbor.get("position"))
    n_vel = _xy(neighbor.get("velocity"))

    ego_len = _safe_float(ego_state.get("length"))
    ego_wid = _safe_float(ego_state.get("width"))
    n_len = _safe_float(neighbor.get("length"))
    n_wid = _safe_float(neighbor.get("width"))

    center_dist = None
    clearance_est = None
    if ego_pos is not None and n_pos is not None:
        center_dist = float(np.linalg.norm(ego_pos - n_pos))
        if None not in (ego_len, ego_wid, n_len, n_wid):
            ego_r = float(np.hypot(ego_len, ego_wid) / 2.0)
            n_r = float(np.hypot(n_len, n_wid) / 2.0)
            clearance_est = center_dist - (ego_r + n_r)

    return {
        "entity_id": neighbor.get("entity_id"),
        "type": neighbor.get("type"),
        "keys": sorted(list(neighbor.keys())),
        "has_position": n_pos is not None,
        "has_velocity": n_vel is not None,
        "has_speed_m_s": neighbor.get("speed_m_s") is not None,
        "has_polygon": neighbor.get("polygon") is not None,
        "speed_m_s": _safe_float(neighbor.get("speed_m_s")),
        "center_dist": center_dist,
        "clearance_est": clearance_est,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--traffic-density", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=default_output_path_str("neighbor_debug.jsonl"))
    parser.add_argument("--max-neighbors-diag", type=int, default=0)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    conf_dir = repo_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "reward=scalar_rulebook",
                "curriculum=stages",
                "seed=42",
                f"reward.rule_margin_log_path={default_output_path_str('debug_rule_margins_from_neighbor_script.jsonl')}",
            ],
        )

    env = build_env(
        cfg,
        env_overrides={
            "map": 5,
            "traffic_density": float(args.traffic_density),
            "start_seed": 10000,
            "num_scenarios": 1,
            "horizon": 500,
        },
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    obs, info = env.reset()

    for t in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        ego_state = info.get("ego_state") if isinstance(info.get("ego_state"), dict) else {}
        neighbors = info.get("neighbors") if isinstance(info.get("neighbors"), list) else []

        neighbors_diag = [_neighbor_diag(ego_state, n) for n in neighbors if isinstance(n, dict)]
        if args.max_neighbors_diag > 0:
            neighbors_diag = neighbors_diag[: args.max_neighbors_diag]

        clearance_values = [
            float(item["clearance_est"])
            for item in neighbors_diag
            if isinstance(item.get("clearance_est"), (int, float))
        ]
        center_dist_values = [
            float(item["center_dist"])
            for item in neighbors_diag
            if isinstance(item.get("center_dist"), (int, float))
        ]

        nearest_neighbor = None
        if neighbors_diag:
            sortable = [
                item
                for item in neighbors_diag
                if isinstance(item.get("clearance_est"), (int, float))
            ]
            if sortable:
                nearest_neighbor = min(sortable, key=lambda item: float(item["clearance_est"]))

        record = {
            "step": info.get("episode_length", t),
            "neighbors_count": len(neighbors),
            "rule_components": dict(info.get("rule_components", {})),
            "rule_input_available": dict(info.get("rule_input_available", {})),
            "rule_input_sources": dict(info.get("rule_input_sources", {})),
            "neighbors_diag": neighbors_diag,
            "min_clearance_est": min(clearance_values) if clearance_values else None,
            "min_center_dist": min(center_dist_values) if center_dist_values else None,
            "nearest_neighbor": nearest_neighbor,
            "first_neighbor_raw": _jsonable(neighbors[0]) if neighbors else None,
            "ego_state_keys": sorted(list(ego_state.keys())),
            "ego_pos": _jsonable(ego_state.get("position")),
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")

        if terminated or truncated:
            break

    env.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
