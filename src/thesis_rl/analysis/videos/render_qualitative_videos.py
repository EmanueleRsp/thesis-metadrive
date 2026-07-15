from __future__ import annotations

import argparse
import csv
from contextlib import suppress
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.agent.adapters.interfaces.base import BaseAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_preprocessor,
    load_planner,
)
from thesis_rl.runtime.execution.seeding import seed_env_spaces, set_global_seed

from thesis_rl.analysis.videos.render_selected_videos import (
    _ensure_omegaconf_now_resolver,
    _resolve_replay_checkpoint,
    _resolve_replay_overrides_for_episode,
    _sanitize_cfg_for_replay,
    _render_topdown_frame,
    _save_gif,
)


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))
    return cleaned.strip("_") or "unknown"


def _render_one_episode(
    *,
    run_dir: Path,
    eval_id: int,
    episode_id: int,
    scenario_seed: int,
    stage_name: str,
    output_path: Path,
) -> None:
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
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _reward, done, truncated, _step_info = env.step(action)
            frame = _render_topdown_frame(env, topdown_cfg)
            if frame is not None:
                frames.append(np.asarray(frame))
    finally:
        if env is not None:
            with suppress(Exception):
                env.close()

    _save_gif(frames, output_path, fps=fps)
    _ = (eval_id, episode_id)  # kept for explicit signature symmetry


def render_qualitative_videos(
    *,
    comparison_root: Path,
) -> None:
    qualitative_dir = comparison_root / "qualitative"
    manifest_path = qualitative_dir / "video_manifest.csv"
    rows, fieldnames = _read_manifest(manifest_path)

    if "render_rel_path" not in fieldnames:
        fieldnames = [*fieldnames, "render_rel_path"]
        for row in rows:
            row["render_rel_path"] = row.get("render_rel_path", "")

    gifs_dir = qualitative_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    for row in rows:
        if str(row.get("selection_status", "")).strip().lower() != "selected":
            skipped += 1
            continue

        run_dir_text = str(row.get("run_dir", "")).strip()
        eval_id_text = str(row.get("eval_id", "")).strip()
        episode_id_text = str(row.get("episode_id", "")).strip()
        scenario_seed_text = str(row.get("scenario_seed", "")).strip()
        if not run_dir_text or not eval_id_text or not episode_id_text or not scenario_seed_text:
            row["selection_status"] = "unavailable"
            row["selection_reason"] = "missing_required_ids"
            skipped += 1
            continue

        run_dir = Path(run_dir_text)
        category = _safe_name(str(row.get("category", "selected")).strip())
        condition_id = _safe_name(str(row.get("condition_id", "condition")).strip())
        seed = _safe_name(str(row.get("seed", "na")).strip())
        eval_id = int(float(eval_id_text))
        episode_id = int(float(episode_id_text))
        scenario_seed = int(float(scenario_seed_text))
        stage_name = str(row.get("stage", "baseline")).strip() or "baseline"

        filename = (
            f"{condition_id}__{category}__seed{seed}__eval{eval_id:04d}__ep{episode_id:04d}.gif"
        )
        output_path = gifs_dir / filename
        rel_path = str(Path("qualitative") / "gifs" / filename)

        try:
            _render_one_episode(
                run_dir=run_dir,
                eval_id=eval_id,
                episode_id=episode_id,
                scenario_seed=scenario_seed,
                stage_name=stage_name,
                output_path=output_path,
            )
            row["render_rel_path"] = rel_path
            row["selection_reason"] = "rendered"
            rendered += 1
            print(f"Rendered qualitative GIF -> {output_path}")
        except Exception as exc:
            row["selection_status"] = "unavailable"
            row["selection_reason"] = f"render_failed:{exc}"
            row["render_rel_path"] = ""
            skipped += 1
            print(f"[qualitative][warn] Failed render for condition={condition_id} category={category}: {exc}")

    _write_manifest(manifest_path, rows, fieldnames)
    print(f"Updated qualitative manifest -> {manifest_path}")
    print(f"Qualitative render summary: rendered={rendered}, skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render qualitative GIFs from comparison video_manifest.csv.")
    parser.add_argument("--comparison-root", required=True)
    args = parser.parse_args()
    render_qualitative_videos(comparison_root=Path(args.comparison_root))


if __name__ == "__main__":
    main()
