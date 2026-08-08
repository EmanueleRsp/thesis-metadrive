from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.eval_artifacts import (
    build_live_final_eval_recorder_factory,
    build_periodic_tracked_subset_recorder_factory,
    maybe_build_periodic_tracked_subset_recorder_factory,
)


class _EvalPlanner:
    def predict(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.0, 0.0], dtype=np.float32), None


class _RenderEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self, **kwargs):
        _ = kwargs
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step += 1
        done = self._step >= 2
        info = {
            "arrive_dest": done,
            "route_completion": 1.0 if done else 0.5,
            "rule_reward_vector": [0.2, 0.1],
            "rule_metadata": {
                "version": "v2",
                "rule_names": ["collision_impact", "dynamic_interaction_safety"],
                "priorities": [0, 1],
                "saturation_ratio_by_rule": {
                    "collision_impact": 0.2,
                    "dynamic_interaction_safety": 0.1,
                }
            },
        }
        return np.array([self._step, self._step], dtype=np.float32), 0.5, done, False, info

    def render(self, mode="topdown", **kwargs):
        _ = (mode, kwargs)
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _ScenarioUidRenderEnv(_RenderEnv):
    """Like ``_RenderEnv`` but stamps ``scenario_uid`` into ``step_info``,
    matching ``thesis_scenario_env.step``'s real behavior (scenario metadata
    is merged into every step's info dict)."""

    def __init__(self, scenario_uid: str) -> None:
        super().__init__()
        self._scenario_uid = scenario_uid

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info = {**info, "scenario_uid": self._scenario_uid}
        return obs, reward, done, truncated, info


def _build_cfg(tmp_path: Path):
    run_dir = tmp_path / "run"
    return OmegaConf.create(
        {
            "seed": 7,
            "experiment": {"eval_deterministic": True},
            "reward": {
                "type": "rulebook",
                "behavior": "scalar_reward",
                "rulebook_config": "selection",
            },
            "video": {
                "fps": 12,
                "topdown": {
                    "window": False,
                    "screen_record": True,
                    "screen_size": [32, 32],
                    "scaling": 2,
                    "semantic_map": False,
                },
            },
            "paths": {
                "run_dir": str(run_dir),
                "videos_dir": str(run_dir / "videos"),
            },
        }
    )


def test_live_eval_recorder_writes_manifest_and_video(tmp_path: Path, monkeypatch) -> None:
    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr("thesis_rl.runtime.io.eval_artifacts.save_gif", _fake_save_gif)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={"traffic_density": 0.1, "horizon": 500, "map": 5},
        eval_id=3,
        eval_type="final",
        scenario_set="test",
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/final.zip",
        checkpoint_type="final",
        checkpoint_global_step=1234,
    )

    recorder = factory({"episode_id": 1, "scenario_seed": 42})
    env = _RenderEnv()
    recorder.record_step(
        env=env,
        step_index=0,
        observation=np.array([0.0, 0.0], dtype=np.float32),
        next_observation=np.array([1.0, 1.0], dtype=np.float32),
        action=np.array([0.0, 0.0], dtype=np.float32),
        reward=0.5,
        done=False,
        truncated=False,
        step_info={
            "route_completion": 0.5,
            "rule_reward_vector": (0.1, 0.0, -0.2, 0.3),
            "rule_metadata": {"version": "v2", "rule_names": ["collision_impact"]},
            "rule_components": {"collision_impact": {"cost": 0.0}},
            "rulebook": {"complete_evaluation": True},
        },
    )
    payload = recorder.finalize_episode(
        episode_metrics={
            "reward": 1.0,
            "episode_length": 2,
            "success": True,
            "collision": False,
            "out_of_road": False,
            "timeout": False,
            "route_completion": 1.0,
            "top_rule_violation_rate": 0.0,
            "error_value": 0.0,
            "violated_rules": "none",
            "violation_pattern": "none",
        }
    )

    assert payload["video_recorded_live"] is True
    assert payload["video_path"] == "videos/final_eval/test/eval_0003/episode_0001.gif"
    assert payload["video_manifest_path"] == (
        "videos/final_eval/test/eval_0003/episode_0001.manifest.json"
    )
    assert payload["trajectory_log_path"] == (
        "videos/final_eval/test/eval_0003/episode_0001.trajectory.jsonl"
    )

    manifest_path = run_dir / str(payload["video_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint_path"] == "checkpoints/final.zip"
    assert manifest["scenario_seed"] == 42
    assert manifest["episode_metrics"]["reward"] == 1.0
    assert "RenderEnv" in manifest["wrappers"]
    assert manifest["trajectory_log_path"] == (
        "videos/final_eval/test/eval_0003/episode_0001.trajectory.jsonl"
    )

    trajectory_path = run_dir / str(payload["trajectory_log_path"])
    rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["t"] == 0
    assert rows[0]["reward"] == 0.5
    assert rows[0]["action"] == [0.0, 0.0]
    assert rows[0]["rule_metadata"]["version"] == "v2"
    assert rows[0]["rule_components"]["collision_impact"]["cost"] == 0.0
    assert rows[0]["rulebook"]["complete_evaluation"] is True


def test_trajectory_log_preserves_termination_diagnostics(tmp_path: Path, monkeypatch) -> None:
    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr(
        "thesis_rl.runtime.io.eval_artifacts.save_gif",
        _fake_save_gif,
    )
    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=3,
        eval_type="final",
        scenario_set="test",
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoint.zip",
        checkpoint_type="final",
        checkpoint_global_step=0,
    )
    recorder = factory({"episode_id": 2, "scenario_seed": 42})
    recorder.record_step(
        env=_RenderEnv(),
        step_index=0,
        observation=np.zeros(2, dtype=np.float32),
        next_observation=np.ones(2, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        reward=0.0,
        done=False,
        truncated=False,
        step_info={
            "crash_sidewalk": False,
            "out_of_road": False,
            "physical_out_of_road": False,
            "geometric_full_footprint_exit": False,
            "geometric_outside_area_m2": 0.0,
            "geometric_ego_area_m2": 8.0,
            "crossed_continuous_line": True,
            "termination_reason": None,
            "route_lateral": 5.0,
            "dist_to_left_side": 3.0,
            "dist_to_right_side": 3.0,
            "on_lane": True,
            "contact_results": ["ROAD_EDGE_BOUNDARY"],
            "terminated": False,
            "truncated": False,
        },
    )
    recorder.finalize_episode(episode_metrics={})
    trajectory_path = run_dir / "videos/final_eval/test/eval_0003/episode_0002.trajectory.jsonl"
    row = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert row["info"]["physical_out_of_road"] is False
    assert row["info"]["geometric_full_footprint_exit"] is False
    assert row["info"]["geometric_outside_area_m2"] == 0.0
    assert row["info"]["crossed_continuous_line"] is True
    assert row["info"]["route_lateral"] == 5.0
    assert row["info"]["dist_to_right_side"] == 3.0
    assert row["info"]["contact_results"] == ["ROAD_EDGE_BOUNDARY"]
    assert row["info"]["terminated"] is False
    assert row["info"]["truncated"] is False


def test_agent_evaluate_returns_live_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr("thesis_rl.runtime.io.eval_artifacts.save_gif", _fake_save_gif)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={"traffic_density": 0.1, "horizon": 500},
        eval_id=1,
        eval_type="final",
        scenario_set="test",
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/final.zip",
        checkpoint_type="final",
        checkpoint_global_step=999,
    )

    agent = Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_EvalPlanner(),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,)),
    )
    metrics = agent.evaluate(
        env=_RenderEnv(),
        n_eval_episodes=1,
        deterministic=True,
        base_seed=100,
        return_episode_metrics=True,
        show_progress=False,
        artifact_recorder_factory=factory,
    )

    per_episode = metrics["per_episode"]
    assert per_episode["video_recorded_live"] == [True]
    assert per_episode["video_path"] == ["videos/final_eval/test/eval_0001/episode_0001.gif"]
    assert per_episode["video_authoritative_path"] == [
        "videos/final_eval/test/eval_0001/episode_0001.gif"
    ]
    assert per_episode["video_manifest_path"] == [
        "videos/final_eval/test/eval_0001/episode_0001.manifest.json"
    ]
    assert per_episode["trajectory_log_path"] == [
        "videos/final_eval/test/eval_0001/episode_0001.trajectory.jsonl"
    ]
    assert {row["rule_name"] for row in metrics["per_rule"]} == {
        "collision_impact",
        "dynamic_interaction_safety",
    }


def test_csv_recorder_schema_keeps_live_video_fields(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    recorder.append_row(
        "eval_episodes.csv",
        {
            "algorithm": "algo",
            "reward_type": "native",
            "reward_behavior": "off",
            "curriculum_name": "disabled",
            "rulebook_config": "none",
            "seed": 1,
            "run_id": "run",
            "eval_id": 1,
            "eval_type": "final",
            "scenario_set": "test",
            "episode_id": 1,
            "stage": "baseline",
            "stage_index": 0,
            "global_step": 10,
            "scenario_seed": 100,
            "scenario_id": "seed_100",
            "deterministic": True,
            "reward": 1.0,
            "video_path": "videos/final_eval/eval_0001/episode_0001.gif",
            "video_authoritative_path": "videos/final_eval/eval_0001/episode_0001.gif",
            "video_manifest_path": "videos/final_eval/eval_0001/episode_0001.manifest.json",
            "trajectory_log_path": None,
            "video_recorded_live": True,
            "replay_warning": None,
        },
    )

    contents = (tmp_path / "eval_episodes.csv").read_text(encoding="utf-8")
    assert "video_authoritative_path" in contents
    assert "video_manifest_path" in contents
    assert "video_recorded_live" in contents


# --- REQ-014/DEC-014 (amended 2026-07-25): periodic tracked-subset GIF
# rendering (`build_periodic_tracked_subset_recorder_factory`). ---


def test_periodic_tracked_subset_recorder_persists_tracked_scenario_uid(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr("thesis_rl.runtime.io.eval_artifacts.save_gif", _fake_save_gif)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_periodic_tracked_subset_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={"traffic_density": 0.1, "horizon": 500},
        eval_id=4,
        global_step=100_000,
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/eval_snapshot.zip",
        checkpoint_type="periodic_validation_snapshot",
        checkpoint_global_step=100_000,
        tracked_scenario_uids=frozenset({"waymo:v1:arm_a:0007"}),
    )

    recorder = factory({"episode_id": 3, "scenario_seed": 55})
    env = _ScenarioUidRenderEnv("waymo:v1:arm_a:0007")
    recorder.record_step(
        env=env,
        step_index=0,
        observation=np.array([0.0, 0.0], dtype=np.float32),
        next_observation=np.array([1.0, 1.0], dtype=np.float32),
        action=np.array([0.0, 0.0], dtype=np.float32),
        reward=0.5,
        done=False,
        truncated=False,
        step_info=env.step(np.array([0.0, 0.0], dtype=np.float32))[4],
    )
    payload = recorder.finalize_episode(episode_metrics={"reward": 1.0})

    assert payload["video_recorded_live"] is True
    expected_prefix = "videos/periodic_eval/step_0100000/eval_0004/waymo:v1:arm_a:0007"
    assert payload["video_path"] == f"{expected_prefix}.gif"
    assert payload["video_manifest_path"] == f"{expected_prefix}.manifest.json"
    assert (run_dir / str(payload["video_path"])).exists()
    manifest = json.loads((run_dir / str(payload["video_manifest_path"])).read_text(encoding="utf-8"))
    assert manifest["scenario_uid"] == "waymo:v1:arm_a:0007"
    assert manifest["eval_type"] == "periodic_tracked_subset"


def test_periodic_tracked_subset_recorder_discards_untracked_scenario_uid(
    tmp_path: Path, monkeypatch
) -> None:
    save_calls: list[Path] = []

    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        save_calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr("thesis_rl.runtime.io.eval_artifacts.save_gif", _fake_save_gif)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_periodic_tracked_subset_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=4,
        global_step=100_000,
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/eval_snapshot.zip",
        checkpoint_type="periodic_validation_snapshot",
        checkpoint_global_step=100_000,
        tracked_scenario_uids=frozenset({"waymo:v1:arm_a:0007"}),
    )

    recorder = factory({"episode_id": 1, "scenario_seed": 1})
    env = _ScenarioUidRenderEnv("waymo:v1:arm_b:9999")  # not in the tracked subset
    recorder.record_step(
        env=env,
        step_index=0,
        observation=np.array([0.0, 0.0], dtype=np.float32),
        next_observation=np.array([1.0, 1.0], dtype=np.float32),
        action=np.array([0.0, 0.0], dtype=np.float32),
        reward=0.5,
        done=False,
        truncated=False,
        step_info=env.step(np.array([0.0, 0.0], dtype=np.float32))[4],
    )
    payload = recorder.finalize_episode(episode_metrics={"reward": 1.0})

    assert payload["video_recorded_live"] is False
    assert payload["video_path"] is None
    assert payload["video_manifest_path"] is None
    assert payload["replay_warning"] == "periodic_tracked_subset_skip:not_in_tracked_subset"
    assert save_calls == []
    # Nothing should have been written to disk for an untracked scenario_uid.
    assert not (run_dir / "videos" / "periodic_eval").exists()


def test_maybe_build_periodic_tracked_subset_recorder_factory_none_when_empty_uids(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.video["enabled"] = True
    cfg.video["mode"] = "live_final_eval"
    run_dir = Path(str(cfg.paths.run_dir))
    factory = maybe_build_periodic_tracked_subset_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=1,
        global_step=100_000,
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/eval_snapshot.zip",
        checkpoint_type="periodic_validation_snapshot",
        checkpoint_global_step=100_000,
        tracked_scenario_uids=(),
    )
    assert factory is None


def test_maybe_build_periodic_tracked_subset_recorder_factory_none_when_video_disabled(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.video["enabled"] = False
    run_dir = Path(str(cfg.paths.run_dir))
    factory = maybe_build_periodic_tracked_subset_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=1,
        global_step=100_000,
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/eval_snapshot.zip",
        checkpoint_type="periodic_validation_snapshot",
        checkpoint_global_step=100_000,
        tracked_scenario_uids=("waymo:v1:arm_a:0007",),
    )
    assert factory is None


def test_maybe_build_periodic_tracked_subset_recorder_factory_independent_of_record_intermediate_evals(
    tmp_path: Path,
) -> None:
    """Regression: the periodic tracked-subset render is a separate,
    DEC-014-approved mechanism and must not depend on the pre-existing
    ``record_intermediate_evals`` flag (unlike the full-panel intermediate
    recording gated by `maybe_build_live_final_eval_recorder_factory`)."""
    cfg = _build_cfg(tmp_path)
    cfg.video["enabled"] = True
    cfg.video["mode"] = "live_final_eval"
    cfg.video["record_intermediate_evals"] = False
    run_dir = Path(str(cfg.paths.run_dir))
    factory = maybe_build_periodic_tracked_subset_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=1,
        global_step=100_000,
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/eval_snapshot.zip",
        checkpoint_type="periodic_validation_snapshot",
        checkpoint_global_step=100_000,
        tracked_scenario_uids=("waymo:v1:arm_a:0007",),
    )
    assert factory is not None


def test_final_eval_recorder_factory_unaffected_by_periodic_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: final-test rendering (`build_live_final_eval_recorder_factory`)
    keeps its `final_eval/<scenario_set>/eval_<id>/...` path convention and
    unconditional full-panel behavior -- it must not pick up the periodic
    tracked-subset output directory or scenario_uid-based filename/filtering
    defaults added for Milestone 9."""

    def _fake_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr("thesis_rl.runtime.io.eval_artifacts.save_gif", _fake_save_gif)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(str(cfg.paths.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config={},
        eval_id=9,
        eval_type="final",
        scenario_set="test",
        stage="baseline",
        stage_index=0,
        checkpoint_path="checkpoints/final.zip",
        checkpoint_type="final",
        checkpoint_global_step=1_500_000,
    )
    recorder = factory({"episode_id": 5, "scenario_seed": 7})
    # An untracked scenario_uid (not filtered: `tracked_scenario_uids` defaults
    # to None for the final path, so every episode still persists).
    env = _ScenarioUidRenderEnv("waymo:v1:arm_z:1234")
    recorder.record_step(
        env=env,
        step_index=0,
        observation=np.array([0.0, 0.0], dtype=np.float32),
        next_observation=np.array([1.0, 1.0], dtype=np.float32),
        action=np.array([0.0, 0.0], dtype=np.float32),
        reward=0.5,
        done=False,
        truncated=False,
        step_info=env.step(np.array([0.0, 0.0], dtype=np.float32))[4],
    )
    payload = recorder.finalize_episode(episode_metrics={"reward": 1.0})

    assert payload["video_recorded_live"] is True
    assert payload["video_path"] == "videos/final_eval/test/eval_0009/episode_0005.gif"
    assert payload["video_manifest_path"] == (
        "videos/final_eval/test/eval_0009/episode_0005.manifest.json"
    )
