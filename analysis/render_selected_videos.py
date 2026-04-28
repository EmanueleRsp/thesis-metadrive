from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.agents.agent import Agent
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.runtime.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_preprocessor,
    load_planner,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing selection file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [dict(item) for item in data if isinstance(item, dict)]


def _find_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir missing: {ckpt_dir}")
    candidates = sorted([p for p in ckpt_dir.glob("*.zip") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No planner checkpoint .zip in {ckpt_dir}")
    return candidates[-1]


def _resolve_eval_overrides_for_stage(cfg: Any, stage_name: str) -> dict[str, Any] | None:
    curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
    if not (curriculum_cfg.enabled and curriculum_cfg.stages):
        return None
    for stage in curriculum_cfg.stages:
        if stage.name == stage_name:
            merged = dict(stage.env)
            merged.update(stage.eval_env)
            return merged
    return None


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for GIF export. Install `pillow`.") from exc

    if not frames:
        raise ValueError("No frames to save.")
    pil_frames = [Image.fromarray(_to_uint8_rgb(frame)) for frame in frames]
    duration_ms = int(max(1, round(1000.0 / max(int(fps), 1))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _update_video_index(index_path: Path, rows: list[dict[str, Any]]) -> None:
    if not index_path.exists():
        return
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
        fieldnames = list(existing[0].keys()) if existing else []
    if not fieldnames:
        return

    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in existing:
        key = (str(row.get("eval_id", "")), str(row.get("episode_id", "")), str(row.get("scenario_seed", "")))
        by_key[key] = row
    for row in rows:
        key = (str(row.get("eval_id", "")), str(row.get("episode_id", "")), str(row.get("scenario_seed", "")))
        if key in by_key:
            for col, value in row.items():
                by_key[key][str(col)] = str(value)

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_key.values())


def _update_eval_episodes_video_paths(eval_path: Path, rows: list[dict[str, Any]]) -> None:
    with eval_path.open("r", encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
        fieldnames = list(data[0].keys()) if data else []
    if not data or "video_path" not in fieldnames:
        return

    updates: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (str(row.get("eval_id", "")), str(row.get("episode_id", "")), str(row.get("scenario_seed", "")))
        updates[key] = str(row.get("video_path", ""))

    for row in data:
        key = (str(row.get("eval_id", "")), str(row.get("episode_id", "")), str(row.get("scenario_seed", "")))
        if key in updates:
            row["video_path"] = updates[key]

    with eval_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def render_selected_videos(run_dir: Path) -> None:
    selection_path = run_dir / "videos" / "metadata" / "video_selection.json"
    index_path = run_dir / "videos" / "metadata" / "video_index.csv"
    eval_episodes_path = run_dir / "csv" / "eval_episodes.csv"
    items = _read_json(selection_path)
    if not items:
        print(f"No selected episodes found in {selection_path}")
        return

    hydra_cfg_path = run_dir / "hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config snapshot: {hydra_cfg_path}")
    cfg = OmegaConf.load(hydra_cfg_path)

    checkpoint_path = _find_checkpoint(run_dir)
    fps = int(cfg.video.get("fps", 20))
    topdown_cfg = cfg.video.get("topdown", {})

    updated_rows: list[dict[str, Any]] = []
    for item in items:
        eval_id = int(item["eval_id"])
        episode_id = int(item["episode_id"])
        scenario_seed = int(item["scenario_seed"])
        stage_name = str(item.get("stage", "baseline"))
        tag = str(item.get("tag", "selected"))

        overrides = _resolve_eval_overrides_for_stage(cfg, stage_name)
        env = build_env(cfg, overrides)
        preprocessor = build_preprocessor(cfg)
        adapter: BaseAdapter = build_adapter(cfg, adapter_space_kwargs(env.action_space))
        planner = load_planner(cfg, checkpoint_path=str(checkpoint_path), env=env)
        agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)
        agent.load_adapter(checkpoint_path=checkpoint_path, strict=True)

        obs, _info = env.reset(seed=scenario_seed)
        done = False
        truncated = False
        frames: list[np.ndarray] = []
        replay_reward_sum = 0.0
        replay_success = False
        replay_collision = False
        replay_out_of_road = False
        replay_route_completion = 0.0
        ep_rule_min_margin: dict[str, float] = {}
        rule_priority: dict[str, int] = {}
        error_priority_base = float(cfg.reward.get("a", 2.01))
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)
            replay_reward_sum += float(reward)
            if isinstance(step_info, dict):
                replay_success = replay_success or bool(step_info.get("arrive_dest", False) or step_info.get("success", False))
                replay_collision = replay_collision or any(
                    bool(step_info.get(key, False))
                    for key in ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "collision")
                )
                replay_out_of_road = replay_out_of_road or bool(step_info.get("out_of_road", False))
                for key in ("route_completion", "route_completion_ratio", "progress"):
                    value = step_info.get(key)
                    if value is not None:
                        replay_route_completion = max(replay_route_completion, float(value))
                        break
                if replay_route_completion <= 0.0 and replay_success:
                    replay_route_completion = 1.0

                meta = step_info.get("rule_metadata")
                margins = step_info.get("rule_reward_vector")
                if isinstance(meta, dict) and isinstance(margins, (list, tuple, np.ndarray)):
                    names = meta.get("rule_names")
                    priorities = meta.get("priorities")
                    if isinstance(names, list) and isinstance(priorities, list):
                        size = min(len(names), len(priorities), len(margins))
                        for idx in range(size):
                            name = str(names[idx])
                            prio = int(priorities[idx])
                            margin = float(margins[idx])
                            rule_priority[name] = prio
                            ep_rule_min_margin[name] = min(ep_rule_min_margin.get(name, float("inf")), margin)

            frame = env.render(
                mode="topdown",
                window=_as_bool(topdown_cfg.get("window"), False),
                screen_record=_as_bool(topdown_cfg.get("screen_record"), False),
                screen_size=tuple(topdown_cfg.get("screen_size", [800, 800])),
                scaling=float(topdown_cfg.get("scaling", 4)),
                semantic_map=_as_bool(topdown_cfg.get("semantic_map"), False),
            )
            if frame is not None:
                frames.append(np.asarray(frame))

        out_name = f"eval_{eval_id:04d}_ep_{episode_id:04d}_{tag}.gif"
        out_path = run_dir / "videos" / "final_eval" / out_name
        _save_gif(frames, out_path, fps=fps)
        env.close()

        replay_error_value = 0.0
        if ep_rule_min_margin:
            p_max = max(rule_priority.get(name, 0) for name in ep_rule_min_margin)
            for name, min_margin in ep_rule_min_margin.items():
                weight = float(error_priority_base) ** float(p_max - int(rule_priority.get(name, 0)))
                replay_error_value += weight * max(0.0, -float(min_margin))

        original_reward = float(item.get("reward", 0.0))
        original_route = float(item.get("route_completion", 0.0))
        original_error = float(item.get("error_value", 0.0))
        original_success = _as_bool(item.get("success"), False)
        original_collision = _as_bool(item.get("collision"), False)
        original_out_of_road = _as_bool(item.get("out_of_road"), False)

        reward_abs_diff = abs(float(replay_reward_sum) - original_reward)
        route_abs_diff = abs(float(replay_route_completion) - original_route)
        error_abs_diff = abs(float(replay_error_value) - original_error)
        success_match = replay_success == original_success
        collision_match = replay_collision == original_collision
        out_of_road_match = replay_out_of_road == original_out_of_road
        replay_match = (
            success_match
            and collision_match
            and out_of_road_match
            and reward_abs_diff <= 1e-2
            and route_abs_diff <= 1e-3
            and error_abs_diff <= 1e-3
        )

        rel_path = str(Path("videos") / "final_eval" / out_name)
        updated_rows.append(
            {
                "eval_id": eval_id,
                "episode_id": episode_id,
                "scenario_seed": scenario_seed,
                "video_path": rel_path,
                "replay_reward": f"{replay_reward_sum:.8f}",
                "reward_abs_diff": f"{reward_abs_diff:.8f}",
                "replay_route_completion": f"{replay_route_completion:.8f}",
                "route_completion_abs_diff": f"{route_abs_diff:.8f}",
                "replay_error_value": f"{replay_error_value:.8f}",
                "error_value_abs_diff": f"{error_abs_diff:.8f}",
                "replay_success": str(replay_success).lower(),
                "success_match": str(success_match).lower(),
                "replay_collision": str(replay_collision).lower(),
                "collision_match": str(collision_match).lower(),
                "replay_out_of_road": str(replay_out_of_road).lower(),
                "out_of_road_match": str(out_of_road_match).lower(),
                "replay_match": str(replay_match).lower(),
            }
        )
        if not replay_match:
            print(
                f"[video][warn] replay mismatch eval={eval_id} ep={episode_id} "
                f"reward_diff={reward_abs_diff:.4g} route_diff={route_abs_diff:.4g} error_diff={error_abs_diff:.4g} "
                f"success_match={success_match} collision_match={collision_match} out_of_road_match={out_of_road_match}"
            )
        print(f"Rendered {rel_path}")

    _update_video_index(index_path, updated_rows)
    _update_eval_episodes_video_paths(eval_episodes_path, updated_rows)
    print(f"Updated video paths in {index_path} and {eval_episodes_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GIF videos for selected episodes via offline replay.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    render_selected_videos(run_dir=Path(args.run_dir))


if __name__ == "__main__":
    main()
