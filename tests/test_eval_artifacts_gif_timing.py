"""step_timing_instrumentation_v1 REQ-003 (docs/implementation/
step_timing_instrumentation_v1_exec_plan.md): GIF render/annotation timing in
``LiveEvalEpisodeRecorder``, isolated from training-loop timing. Follows the
same direct-instantiation pattern as ``tests/test_eval_artifacts_ego_trail.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from thesis_rl.runtime.io import eval_artifacts


def _make_recorder(tmp_path: Path) -> eval_artifacts.LiveEvalEpisodeRecorder:
    return eval_artifacts.LiveEvalEpisodeRecorder(
        run_dir=tmp_path,
        videos_dir=tmp_path / "videos",
        eval_id=0,
        episode_id=0,
        fps=20,
        topdown_cfg={"draw_ego_trail": False},
        manifest_payload={},
        save_manifest=False,
        save_trajectory_log=False,
    )


def _record_one_step(recorder: eval_artifacts.LiveEvalEpisodeRecorder) -> None:
    recorder.record_step(
        env=object(),
        step_index=0,
        observation=None,
        next_observation=None,
        action=np.zeros(2),
        reward=0.0,
        done=False,
        truncated=False,
        step_info={},
    )


def test_gif_render_seconds_accumulates_across_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _slow_render(env, cfg):
        _ = (env, cfg)
        time.sleep(0.001)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(eval_artifacts, "render_topdown_frame", _slow_render)

    recorder = _make_recorder(tmp_path)
    assert recorder._gif_render_seconds == 0.0
    _record_one_step(recorder)
    after_one_step = recorder._gif_render_seconds
    assert after_one_step > 0.0
    _record_one_step(recorder)
    assert recorder._gif_render_seconds > after_one_step


def test_gif_render_seconds_includes_encode_time_in_finalize_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_artifacts,
        "render_topdown_frame",
        lambda env, cfg: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    def _slow_save_gif(frames, output_path: Path, fps: int) -> None:
        _ = (frames, fps)
        time.sleep(0.001)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")

    monkeypatch.setattr(eval_artifacts, "save_gif", _slow_save_gif)

    recorder = _make_recorder(tmp_path)
    _record_one_step(recorder)
    before_finalize = recorder._gif_render_seconds

    payload = recorder.finalize_episode(episode_metrics={})

    assert payload["gif_render_seconds"] > before_finalize


def test_gif_render_seconds_present_but_negligible_when_no_frames_rendered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(eval_artifacts, "render_topdown_frame", lambda env, cfg: None)

    recorder = _make_recorder(tmp_path)
    _record_one_step(recorder)

    payload = recorder.finalize_episode(episode_metrics={})
    # No frame means the diagnostic/annotation branch never ran, and there is
    # nothing to encode in `finalize_episode`, so only the negligible
    # try-block overhead of `record_step` itself is accumulated.
    assert 0.0 <= payload["gif_render_seconds"] < 0.05


def test_new_episode_instance_starts_with_zero_gif_render_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_artifacts,
        "render_topdown_frame",
        lambda env, cfg: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    first_episode = _make_recorder(tmp_path)
    _record_one_step(first_episode)
    assert first_episode._gif_render_seconds > 0.0

    second_episode = _make_recorder(tmp_path)
    assert second_episode._gif_render_seconds == 0.0
