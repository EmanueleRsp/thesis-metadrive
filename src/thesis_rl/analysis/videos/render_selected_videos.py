from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from contextlib import suppress

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.agent.adapters.interfaces.base import BaseAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_preprocessor,
    load_planner,
)
from thesis_rl.runtime.execution.seeding import seed_env_spaces, set_global_seed


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.floating)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    # Common CSV encodings from float/bool pipelines: "1.0", "0.0"
    try:
        return float(text) != 0.0
    except ValueError:
        return default


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


def _resolve_replay_checkpoint(run_dir: Path, cfg: Any) -> Path:
    replay_spec = str(cfg.video.get("replay_checkpoint", "final")).strip()
    if replay_spec == "final":
        ckpt = run_dir / "checkpoints" / "final.zip"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"Configured replay checkpoint not found: {ckpt}")
    if replay_spec == "latest":
        ckpt = run_dir / "checkpoints" / "latest.zip"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"Configured replay checkpoint not found: {ckpt}")
    if replay_spec == "best_lexicographic":
        ckpt = run_dir / "checkpoints" / "best" / "best_lexicographic.zip"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"Configured replay checkpoint not found: {ckpt}")

    explicit = Path(replay_spec)
    if not explicit.is_absolute():
        explicit = run_dir / replay_spec
    if explicit.exists():
        return explicit
    raise FileNotFoundError(f"Configured replay checkpoint path not found: {explicit}")


def _resolve_eval_overrides_for_stage(cfg: Any, stage_name: str) -> dict[str, Any] | None:
    curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
    if not (curriculum_cfg.enabled and curriculum_cfg.is_staged and curriculum_cfg.staged.stages):
        return None
    for stage in curriculum_cfg.staged.stages:
        if stage.name == stage_name:
            merged = dict(stage.env)
            merged.update(stage.eval_env)
            return merged
    return None


def _ensure_omegaconf_now_resolver() -> None:
    """Register a lightweight `now` resolver for standalone replay scripts.

    Hydra registers `now` during app bootstrap, but these replay scripts are
    executed directly via `python -m ...` and may load unresolved `${now:...}`
    interpolations from saved Hydra configs.
    """
    try:
        has_now = OmegaConf.has_resolver("now")
    except Exception:
        has_now = False
    if has_now:
        return

    def _now(pattern: str = "%Y%m%d_%H%M%S") -> str:
        return datetime.now().strftime(str(pattern))

    # Support both old and new OmegaConf APIs.
    try:
        OmegaConf.register_new_resolver("now", _now)  # type: ignore[attr-defined]
    except Exception:
        try:
            OmegaConf.register_resolver("now", _now)  # type: ignore[attr-defined]
        except Exception:
            pass


def _sanitize_cfg_for_replay(cfg: Any, run_dir: Path) -> None:
    """Patch replay-only fields to avoid resolver/side-effect issues."""
    # Avoid unresolved `${now:...}` chain through reward.rule_margin_log_path.
    with suppress(Exception):
        OmegaConf.update(cfg, "reward.rule_margin_log_path", None, force_add=True)

    # Provide concrete paths so any downstream interpolation on `paths.*` is safe.
    with suppress(Exception):
        OmegaConf.update(cfg, "paths.run_dir", str(run_dir), force_add=True)
        OmegaConf.update(cfg, "paths.logs_dir", str(run_dir / "logs"), force_add=True)
        OmegaConf.update(cfg, "paths.checkpoints_dir", str(run_dir / "checkpoints"), force_add=True)
        OmegaConf.update(cfg, "paths.videos_dir", str(run_dir / "videos"), force_add=True)
        OmegaConf.update(cfg, "paths.csv_dir", str(run_dir / "csv"), force_add=True)
        OmegaConf.update(cfg, "paths.artifacts_dir", str(run_dir / "artifacts"), force_add=True)


def _render_topdown_frame(env: Any, topdown_cfg: Any) -> Any:
    """Render a topdown frame from the base env (strict replay path)."""
    kwargs = {
        "window": _as_bool(topdown_cfg.get("window"), False),
        "screen_record": _as_bool(topdown_cfg.get("screen_record"), False),
        "screen_size": tuple(topdown_cfg.get("screen_size", [800, 800])),
        "scaling": float(topdown_cfg.get("scaling", 4)),
        "semantic_map": _as_bool(topdown_cfg.get("semantic_map"), False),
    }
    base_env = getattr(env, "unwrapped", env)
    try:
        return base_env.render(mode="topdown", **kwargs)
    except TypeError as exc:
        # Some MetaDrive/Gym versions accept kwargs but not `mode=...`.
        if "unexpected keyword argument 'mode'" in str(exc):
            return base_env.render(**kwargs)
        raise


def _resolve_replay_overrides_for_episode(
    *,
    cfg: Any,
    stage_name: str,
    scenario_seed: int,
    episode_id: int,
    final_eval_episodes: int,
) -> dict[str, Any]:
    stage_overrides = _resolve_eval_overrides_for_stage(cfg, stage_name) or {}
    overrides = dict(stage_overrides)
    # Reconstruct the same evaluation window used by final eval:
    # scenario_seed = eval_base_seed + (episode_id - 1)
    eval_base_seed = int(scenario_seed) - max(int(episode_id) - 1, 0)
    configured_count = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
    overrides["start_seed"] = int(eval_base_seed)
    overrides["num_scenarios"] = max(configured_count, int(final_eval_episodes))
    return overrides


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


def _annotate_mismatch_frames(
    frames: list[np.ndarray],
    *,
    lines: list[str],
) -> list[np.ndarray]:
    """Overlay a warning box on frames when replay does not perfectly match."""
    if not frames:
        return frames
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frames

    font = ImageFont.load_default()
    out: list[np.ndarray] = []
    for frame in frames:
        img = Image.fromarray(_to_uint8_rgb(frame)).convert("RGB")
        draw = ImageDraw.Draw(img)
        line_h = 14
        pad = 8
        box_w = 540
        box_h = pad * 2 + line_h * len(lines)
        draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0))
        draw.rectangle((8, 8, 8 + box_w, 8 + box_h), outline=(255, 80, 80), width=2)
        y = 8 + pad
        for line in lines:
            draw.text((16, y), line, fill=(255, 220, 220), font=font)
            y += line_h
        out.append(np.asarray(img))
    return out


def _fmt_num(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _annotate_telemetry_frame(
    frame: np.ndarray,
    *,
    lines: list[str],
) -> np.ndarray:
    """Overlay a compact telemetry panel on one frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return np.asarray(frame)

    img = Image.fromarray(_to_uint8_rgb(frame)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    line_h = 14
    pad = 8
    box_w = 500
    box_h = pad * 2 + line_h * len(lines)
    x0 = 8
    y0 = max(8, img.height - box_h - 8)
    x1 = x0 + box_w
    y1 = y0 + box_h
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    draw.rectangle((x0, y0, x1, y1), outline=(100, 180, 255), width=2)
    y = y0 + pad
    for line in lines:
        draw.text((x0 + 8, y), line, fill=(220, 240, 255), font=font)
        y += line_h
    return np.asarray(img)


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
    _ensure_omegaconf_now_resolver()
    cfg = OmegaConf.load(hydra_cfg_path)
    _sanitize_cfg_for_replay(cfg, run_dir)
    final_eval_episodes = int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes))
    run_seed = int(cfg.get("seed", 0))
    set_global_seed(run_seed)

    checkpoint_path = _resolve_replay_checkpoint(run_dir, cfg)
    fps = int(cfg.video.get("fps", 20))
    topdown_cfg = cfg.video.get("topdown", {})

    updated_rows: list[dict[str, Any]] = []
    for item in items:
        eval_id = int(item["eval_id"])
        episode_id = int(item["episode_id"])
        scenario_seed = int(item["scenario_seed"])
        stage_name = str(item.get("stage", "baseline"))
        tag = str(item.get("tag", "selected"))
        env = None
        try:
            overrides = _resolve_replay_overrides_for_episode(
                cfg=cfg,
                stage_name=stage_name,
                scenario_seed=scenario_seed,
                episode_id=episode_id,
                final_eval_episodes=final_eval_episodes,
            )
            env = build_env(cfg, overrides)
            # Mirror final-eval seeding path from training (`run_seed + 500_000`).
            seed_env_spaces(env, run_seed + 500_000)
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
            replay_env_reward_sum = 0.0
            replay_scalar_rule_reward_sum = 0.0
            replay_hybrid_reward_sum = 0.0
            has_scalar_rule_reward = False
            has_hybrid_reward = False
            replay_success = False
            replay_collision = False
            replay_out_of_road = False
            replay_route_completion = 0.0
            ep_rule_min_margin: dict[str, float] = {}
            rule_priority: dict[str, int] = {}
            error_priority_base = float(cfg.reward.get("a", 2.01))
            step_count = 0
            while not (done or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, step_info = env.step(action)
                step_count += 1
                replay_reward_sum += float(reward)
                env_reward_step: float | None = None
                scalar_rule_step: float | None = None
                hybrid_step: float | None = None
                speed_kmh: float | None = None
                top_rule_name: str | None = None
                top_rule_margin: float | None = None
                if isinstance(step_info, dict):
                    env_reward_val = step_info.get("env_reward")
                    if isinstance(env_reward_val, (int, float, np.floating)):
                        env_reward_step = float(env_reward_val)
                    else:
                        env_reward_step = float(reward)
                    replay_env_reward_sum += float(env_reward_step)

                    scalar_rule_val = step_info.get("scalar_rule_reward")
                    if isinstance(scalar_rule_val, (int, float, np.floating)):
                        scalar_rule_step = float(scalar_rule_val)
                        replay_scalar_rule_reward_sum += float(scalar_rule_val)
                        has_scalar_rule_reward = True

                    hybrid_val = step_info.get("hybrid_reward")
                    if isinstance(hybrid_val, (int, float, np.floating)):
                        hybrid_step = float(hybrid_val)
                        replay_hybrid_reward_sum += float(hybrid_val)
                        has_hybrid_reward = True

                    speed_val = step_info.get("velocity")
                    if isinstance(speed_val, (int, float, np.floating)):
                        speed_kmh = float(speed_val)

                    replay_success = replay_success or bool(
                        step_info.get("arrive_dest", False) or step_info.get("success", False)
                    )
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
                            if size > 0:
                                top_rule_name = str(names[0])
                                top_rule_margin = float(margins[0])
                            for idx in range(size):
                                name = str(names[idx])
                                prio = int(priorities[idx])
                                margin = float(margins[idx])
                                rule_priority[name] = prio
                                ep_rule_min_margin[name] = min(ep_rule_min_margin.get(name, float("inf")), margin)
                else:
                    replay_env_reward_sum += float(reward)

                frame = _render_topdown_frame(env, topdown_cfg)
                if frame is not None:
                    line1 = (
                        f"step={step_count} sel_step={_fmt_num(reward)} sel_sum={_fmt_num(replay_reward_sum)} "
                        f"env_sum={_fmt_num(replay_env_reward_sum)}"
                    )
                    line2 = (
                        f"route={_fmt_num(replay_route_completion)} speed_kmh={_fmt_num(speed_kmh)} "
                        f"succ={replay_success} coll={replay_collision} oor={replay_out_of_road}"
                    )
                    lines = [line1, line2]
                    if has_scalar_rule_reward or scalar_rule_step is not None:
                        lines.append(
                            f"scalar_step={_fmt_num(scalar_rule_step)} scalar_sum={_fmt_num(replay_scalar_rule_reward_sum)}"
                        )
                    if has_hybrid_reward or hybrid_step is not None:
                        lines.append(
                            f"hybrid_step={_fmt_num(hybrid_step)} hybrid_sum={_fmt_num(replay_hybrid_reward_sum)}"
                        )
                    if top_rule_name is not None:
                        lines.append(f"top_rule={top_rule_name} margin={_fmt_num(top_rule_margin)}")
                    frames.append(_annotate_telemetry_frame(np.asarray(frame), lines=lines))

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

            out_name = f"eval_{eval_id:04d}_ep_{episode_id:04d}_{tag}.gif"
            out_path = run_dir / "videos" / "final_eval" / out_name
            if not replay_match:
                frames = _annotate_mismatch_frames(
                    frames,
                    lines=[
                        "REPLAY MISMATCH",
                        f"eval={eval_id} ep={episode_id} seed={scenario_seed}",
                        f"dReward={reward_abs_diff:.4g} dRoute={route_abs_diff:.4g} dErr={error_abs_diff:.4g}",
                        f"success={success_match} collision={collision_match} out_of_road={out_of_road_match}",
                    ],
                )
            _save_gif(frames, out_path, fps=fps)

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
        except Exception as exc:
            print(
                f"[video][warn] Failed render for eval={eval_id} ep={episode_id} "
                f"seed={scenario_seed} stage={stage_name}: {exc}"
            )
        finally:
            if env is not None:
                with suppress(Exception):
                    env.close()

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
