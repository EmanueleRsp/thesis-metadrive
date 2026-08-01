"""REQ-002 (video overlay v1, 2026-07-31): ego-trail accumulation lifecycle
in ``LiveEvalEpisodeRecorder``. One recorder instance is created per episode
by both the final-eval and periodic-tracked-subset factories, so the trail
must accumulate across steps within one instance and start empty on a new
instance -- there is no explicit reset call to test separately."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis_rl.runtime.io import eval_artifacts


def _make_recorder(
    tmp_path: Path, *, draw_ego_trail: bool = True
) -> eval_artifacts.LiveEvalEpisodeRecorder:
    return eval_artifacts.LiveEvalEpisodeRecorder(
        run_dir=tmp_path,
        videos_dir=tmp_path / "videos",
        eval_id=0,
        episode_id=0,
        fps=20,
        topdown_cfg={"draw_ego_trail": draw_ego_trail},
        manifest_payload={},
        save_manifest=False,
        save_trajectory_log=False,
    )


def _record_one_step(
    recorder: eval_artifacts.LiveEvalEpisodeRecorder, *, position: tuple[float, float]
) -> None:
    recorder.record_step(
        env=object(),
        step_index=0,
        observation=None,
        next_observation=None,
        action=np.zeros(2),
        reward=0.0,
        done=False,
        truncated=False,
        step_info={"ego_state": {"position": position}},
    )


def test_ego_trail_accumulates_across_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_artifacts, "render_topdown_frame", lambda env, cfg: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    recorder = _make_recorder(tmp_path)
    for position in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]:
        _record_one_step(recorder, position=position)
    assert recorder._ego_trail_world == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]


def test_new_episode_instance_starts_with_empty_trail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_artifacts, "render_topdown_frame", lambda env, cfg: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    first_episode = _make_recorder(tmp_path)
    _record_one_step(first_episode, position=(9.0, 9.0))
    assert first_episode._ego_trail_world == [(9.0, 9.0)]

    second_episode = _make_recorder(tmp_path)
    assert second_episode._ego_trail_world == []


def test_ego_trail_disabled_by_config_does_not_accumulate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_artifacts, "render_topdown_frame", lambda env, cfg: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    recorder = _make_recorder(tmp_path, draw_ego_trail=False)
    _record_one_step(recorder, position=(1.0, 1.0))
    assert recorder._ego_trail_world == []
