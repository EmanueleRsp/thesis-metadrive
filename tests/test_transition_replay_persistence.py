from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_rl.runtime.loops.train_loop import _validate_checkpoint_pair


def test_checkpoint_pair_validation_requires_matching_replay_artifact(tmp_path: Path) -> None:
    pair_path = tmp_path / "final_checkpoint_pair.json"
    replay_path = tmp_path / "final_replay_buffer.pkl"
    replay_path.write_bytes(b"replay")
    pair_path.write_text(
        json.dumps(
            {
                "checkpoint_id": "checkpoint-1",
                "training_timestep": 12,
                "replay_segment_id": 0,
                "model_path": "final.zip",
                "replay_path": "final_replay_buffer.pkl",
            }
        ),
        encoding="utf-8",
    )

    payload = _validate_checkpoint_pair(
        pair_path,
        checkpoint_name="final",
        replay_path=replay_path,
        training_timestep=12,
    )

    assert payload["checkpoint_id"] == "checkpoint-1"


def test_checkpoint_pair_validation_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pair manifest"):
        _validate_checkpoint_pair(
            tmp_path / "missing.json",
            checkpoint_name="final",
            replay_path=tmp_path / "final_replay_buffer.pkl",
        )
