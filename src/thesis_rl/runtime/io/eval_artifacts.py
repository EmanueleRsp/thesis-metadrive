from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_rl.runtime.io.metadata import get_git_commit
from thesis_rl.runtime.io.video_utils import render_topdown_frame, save_gif


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _metadrive_version() -> str:
    try:
        import metadrive  # type: ignore

        version = getattr(metadrive, "__version__", None)
        return str(version) if version is not None else "unknown"
    except Exception:
        return "unknown"


def _wrapper_stack(env: Any) -> list[str]:
    stack: list[str] = []
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        stack.append(type(current).__name__.lstrip("_"))
        current = getattr(current, "env", None)
    return stack


def _compact_step_info(step_info: Any) -> dict[str, Any] | None:
    if not isinstance(step_info, dict):
        return None
    subset: dict[str, Any] = {}
    scalar_keys = (
        "env_reward",
        "scalar_rule_reward",
        "hybrid_reward",
        "route_completion",
        "route_completion_ratio",
        "progress",
        "arrive_dest",
        "success",
        "crash_vehicle",
        "crash_sidewalk",
        "collision",
        "out_of_road",
        "physical_out_of_road",
        "crossed_continuous_line",
        "termination_reason",
        "route_lateral",
        "dist_to_left_side",
        "dist_to_right_side",
        "on_lane",
        "contact_results",
        "geometric_full_footprint_exit",
        "geometric_outside_area_m2",
        "geometric_ego_area_m2",
        "terminated",
        "truncated",
        "timeout",
    )
    for key in scalar_keys:
        if key in step_info:
            subset[key] = _json_safe(step_info.get(key))
    for key in (
        "ego_state",
        "neighbors",
        "rule_reward_vector",
        "rule_metadata",
        "rule_components",
        "rulebook",
    ):
        if key in step_info:
            subset[key] = _json_safe(step_info.get(key))
    return subset


def _trajectory_row(
    *,
    step_index: int,
    observation: Any,
    next_observation: Any,
    action: Any,
    reward: float,
    done: bool,
    truncated: bool,
    step_info: Any,
) -> dict[str, Any]:
    compact_info = _compact_step_info(step_info)
    row: dict[str, Any] = {
        "t": int(step_index),
        "observation": _json_safe(observation),
        "action": _json_safe(action),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "next_observation": _json_safe(next_observation),
        "info": compact_info,
    }
    if isinstance(compact_info, dict):
        for key in (
            "env_reward",
            "scalar_rule_reward",
            "hybrid_reward",
            "route_completion",
            "ego_state",
            "neighbors",
            "rule_reward_vector",
            "rule_metadata",
            "rule_components",
            "rulebook",
        ):
            if key in compact_info:
                row[key] = compact_info[key]
        row["events"] = {
            "success": bool(compact_info.get("arrive_dest") or compact_info.get("success")),
            "collision": bool(compact_info.get("crash_vehicle") or compact_info.get("collision")),
            "out_of_road": bool(compact_info.get("out_of_road")),
        }
    return row


@dataclass(slots=True)
class NoOpEpisodeArtifactRecorder:
    warning: str | None = None

    def record_step(
        self,
        *,
        env: Any,
        step_index: int,
        observation: Any,
        next_observation: Any,
        action: np.ndarray,
        reward: float,
        done: bool,
        truncated: bool,
        step_info: Any,
    ) -> None:
        _ = (env, step_index, observation, next_observation, action, reward, done, truncated, step_info)

    def finalize_episode(self, *, episode_metrics: dict[str, Any]) -> dict[str, Any]:
        _ = episode_metrics
        return {
            "video_path": None,
            "video_authoritative_path": None,
            "video_manifest_path": None,
            "trajectory_log_path": None,
            "video_recorded_live": False,
            "replay_warning": self.warning,
        }


class LiveEvalEpisodeRecorder:
    def __init__(
        self,
        *,
        run_dir: Path,
        videos_dir: Path,
        eval_id: int,
        episode_id: int,
        fps: int,
        topdown_cfg: Any,
        manifest_payload: dict[str, Any],
        save_manifest: bool,
        save_trajectory_log: bool,
    ) -> None:
        self._run_dir = run_dir
        self._videos_dir = videos_dir
        self._eval_id = int(eval_id)
        self._episode_id = int(episode_id)
        self._fps = int(fps)
        self._topdown_cfg = topdown_cfg
        self._manifest_payload = dict(manifest_payload)
        self._save_manifest = bool(save_manifest)
        self._save_trajectory_log = bool(save_trajectory_log)
        self._frames: list[np.ndarray] = []
        self._trajectory_rows: list[dict[str, Any]] = []
        self._warning: str | None = None

    def record_step(
        self,
        *,
        env: Any,
        step_index: int,
        observation: Any,
        next_observation: Any,
        action: np.ndarray,
        reward: float,
        done: bool,
        truncated: bool,
        step_info: Any,
    ) -> None:
        self._trajectory_rows.append(
            _trajectory_row(
                step_index=step_index,
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                truncated=truncated,
                step_info=step_info,
            )
        )
        if self._warning is not None:
            return
        try:
            if self._manifest_payload.get("wrappers") is None:
                self._manifest_payload["wrappers"] = _wrapper_stack(env)
            frame = render_topdown_frame(env, self._topdown_cfg)
            if frame is not None:
                self._frames.append(np.asarray(frame))
        except Exception as exc:
            self._warning = f"live_record_render_failed:{exc}"
            self._frames.clear()

    def finalize_episode(self, *, episode_metrics: dict[str, Any]) -> dict[str, Any]:
        eval_dir = self._videos_dir / "final_eval" / f"eval_{self._eval_id:04d}"
        video_path = eval_dir / f"episode_{self._episode_id:04d}.gif"
        manifest_path = eval_dir / f"episode_{self._episode_id:04d}.manifest.json"
        trajectory_path = eval_dir / f"episode_{self._episode_id:04d}.trajectory.jsonl"

        video_rel = None
        recorded_live = False
        if self._warning is None:
            try:
                save_gif(self._frames, video_path, fps=self._fps)
                video_rel = str(video_path.relative_to(self._run_dir)).replace("\\", "/")
                recorded_live = True
            except Exception as exc:
                self._warning = f"live_record_save_failed:{exc}"

        trajectory_rel = None
        if self._save_trajectory_log:
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            with trajectory_path.open("w", encoding="utf-8") as handle:
                for row in self._trajectory_rows:
                    handle.write(json.dumps(_json_safe(row), ensure_ascii=False))
                    handle.write("\n")
            trajectory_rel = str(trajectory_path.relative_to(self._run_dir)).replace("\\", "/")

        manifest_rel = None
        if self._save_manifest:
            manifest = {
                "schema_version": 1,
                **self._manifest_payload,
                "video_path": video_rel,
                "trajectory_log_path": trajectory_rel,
                "video_recorded_live": recorded_live,
                "replay_warning": self._warning,
                "git_commit": get_git_commit(),
                "metadrive_version": _metadrive_version(),
                "episode_metrics": _json_safe(episode_metrics),
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
            manifest_rel = str(manifest_path.relative_to(self._run_dir)).replace("\\", "/")

        return {
            "video_path": video_rel,
            "video_authoritative_path": video_rel,
            "video_manifest_path": manifest_rel,
            "trajectory_log_path": trajectory_rel,
            "video_recorded_live": recorded_live,
            "replay_warning": self._warning,
        }


def build_live_final_eval_recorder_factory(
    *,
    cfg: Any,
    run_dir: Path,
    resolved_env_config: dict[str, Any],
    eval_id: int,
    eval_type: str,
    scenario_set: str,
    stage: str,
    stage_index: int,
    checkpoint_path: str,
    checkpoint_type: str,
    checkpoint_global_step: int,
) -> Any:
    fps = int(cfg.video.get("fps", 20))
    topdown_cfg = cfg.video.get("topdown", {})
    max_videos = max(int(cfg.video.get("max_final_videos", 0)), 0)
    videos_dir = Path(str(cfg.paths.videos_dir))
    save_manifest = bool(cfg.video.get("save_manifest", True))
    save_trajectory_log = bool(cfg.video.get("save_trajectory_log", True))

    base_manifest = {
        "run_id": Path(str(cfg.paths.run_dir)).name,
        "eval_id": int(eval_id),
        "eval_type": str(eval_type),
        "scenario_set": str(scenario_set),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_type": str(checkpoint_type),
        "checkpoint_global_step": int(checkpoint_global_step),
        "seed": int(cfg.seed),
        "deterministic": bool(cfg.experiment.eval_deterministic),
        "curriculum_stage": str(stage),
        "stage_index": int(stage_index),
        "env_config_resolved": _json_safe(resolved_env_config),
        "map_config": _json_safe(resolved_env_config.get("map_config")),
        "traffic_density": _json_safe(resolved_env_config.get("traffic_density")),
        "traffic_mode": _json_safe(
            resolved_env_config.get("traffic_mode", resolved_env_config.get("traffic_vehicle_config"))
        ),
        "termination_flags": _json_safe(
            {
                "horizon": resolved_env_config.get("horizon"),
                "truncate_as_terminate": resolved_env_config.get("truncate_as_terminate"),
            }
        ),
        "reward_type": str(cfg.reward.type),
        "reward_behavior": str(cfg.reward.behavior),
        "rulebook_config": str(cfg.reward.get("rulebook_config", "none")),
        "wrappers": None,
    }

    def factory(episode_ctx: dict[str, Any]) -> Any:
        episode_id = int(episode_ctx["episode_id"])
        scenario_seed = episode_ctx.get("scenario_seed")
        scenario_uid = episode_ctx.get("scenario_uid")
        manifest_payload = {
            **base_manifest,
            "episode_id": episode_id,
            "scenario_seed": int(scenario_seed) if scenario_seed is not None else None,
            "scenario_id": f"seed_{scenario_seed}" if scenario_seed is not None else None,
            "scenario_uid": str(scenario_uid) if scenario_uid is not None else None,
        }
        if max_videos > 0 and episode_id > max_videos:
            return NoOpEpisodeArtifactRecorder(warning="live_record_skipped:max_final_videos_limit")
        return LiveEvalEpisodeRecorder(
            run_dir=run_dir,
            videos_dir=videos_dir,
            eval_id=eval_id,
            episode_id=episode_id,
            fps=fps,
            topdown_cfg=topdown_cfg,
            manifest_payload=manifest_payload,
            save_manifest=save_manifest,
            save_trajectory_log=save_trajectory_log,
        )

    return factory


def maybe_build_live_final_eval_recorder_factory(
    *,
    cfg: Any,
    run_dir: Path,
    resolved_env_config: dict[str, Any],
    eval_id: int,
    eval_type: str,
    scenario_set: str,
    stage: str,
    stage_index: int,
    checkpoint_path: str,
    checkpoint_type: str,
    checkpoint_global_step: int,
) -> Any:
    if not bool(cfg.video.get("enabled", False)):
        return None
    if str(cfg.video.get("mode", "offline_replay")).strip().lower() != "live_final_eval":
        return None
    if eval_type == "final" and not bool(cfg.video.get("record_final_eval", True)):
        return None
    if eval_type != "final" and not bool(cfg.video.get("record_intermediate_evals", False)):
        return None
    return build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config=resolved_env_config,
        eval_id=eval_id,
        eval_type=eval_type,
        scenario_set=scenario_set,
        stage=stage,
        stage_index=stage_index,
        checkpoint_path=checkpoint_path,
        checkpoint_type=checkpoint_type,
        checkpoint_global_step=checkpoint_global_step,
    )
