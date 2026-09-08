"""Audit 2026-09-06, block A3: checkpoint provenance.

The checkpoints written when an asynchronous validation completes must be copies
of the snapshot that was evaluated, not a fresh save of the live learner.

Block A8's half of this file — `C19`, refusing to resume an off-policy learner
without its persisted replay buffer — was **superseded** by `RESUME-ABRUPT-001`,
which replaced the guard with `classify_replay_resume`: that function makes the
same refusal, names the exact override that would proceed, and additionally
distinguishes a partially committed pair and an explicitly accepted empty
segment. `tests/test_resume_snapshot.py` covers every branch of it, so the
contract is enforced more strictly than before and nothing here was lost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.runtime.loops.train_loop import _copy_checkpoint_snapshot


def test_copy_checkpoint_snapshot_copies_zip_and_sidecars_but_not_the_job_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoints" / "eval_000001_step_000000025000"
    source.parent.mkdir(parents=True)
    source.with_suffix(".zip").write_bytes(b"weights-at-25000")
    (source.parent / f"{source.name}.adapter.pt").write_bytes(b"adapter")
    (source.parent / f"{source.name}.reward_semantics.json").write_text("{}", encoding="utf-8")
    (source.parent / f"{source.name}.manifest.json").write_text("{}", encoding="utf-8")
    source.with_suffix(".json").write_text('{"eval_id": 1}', encoding="utf-8")
    # A sibling snapshot with a longer name must not be swept in.
    (source.parent / f"{source.name}0.zip").write_bytes(b"other")

    target = tmp_path / "checkpoints" / "best" / "best_lexicographic"
    _copy_checkpoint_snapshot(source, target)

    assert target.with_suffix(".zip").read_bytes() == b"weights-at-25000"
    assert (target.parent / "best_lexicographic.adapter.pt").read_bytes() == b"adapter"
    assert (target.parent / "best_lexicographic.reward_semantics.json").exists()
    assert (target.parent / "best_lexicographic.manifest.json").exists()
    assert not target.with_suffix(".json").exists()
    assert sorted(path.name for path in target.parent.iterdir()) == [
        "best_lexicographic.adapter.pt",
        "best_lexicographic.manifest.json",
        "best_lexicographic.reward_semantics.json",
        "best_lexicographic.zip",
    ]


def test_copy_checkpoint_snapshot_requires_the_snapshot_zip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _copy_checkpoint_snapshot(tmp_path / "missing", tmp_path / "target")

