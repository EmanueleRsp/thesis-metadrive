"""`RESUME-ABRUPT-001` unit matrix: crash-safe snapshots and resume classification."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.catalog_state import ScenarioCatalogVisitState
from thesis_rl.curriculum.scenario_acl.driver import (
    ScenarioAclDriverPaths,
    _acl_state_dir_for_checkpoint,
    _write_acl_resume_snapshot,
    _write_acl_state_files,
)
from thesis_rl.runtime.io.atomic import atomic_publish, atomic_write_text, temporary_sibling
from thesis_rl.runtime.io.resume_snapshot import (
    assert_snapshot_model_consistent,
    assert_snapshot_seed_consistent,
    classify_replay_resume,
    periodic_snapshot_due,
    planner_trained_timesteps,
    prune_old_periodic_checkpoints,
    prune_replay_snapshots,
    remove_replay_snapshots_after_final,
    resolve_resume_checkpoint_name,
    resume_artifact_paths,
    write_checkpoint_pair,
)
from thesis_rl.runtime.loops.train_loop import _best_keys_payload, _restore_best_key
from thesis_rl.runtime.signals import install_sigterm_as_keyboard_interrupt

# ---------------------------------------------------------------------------
# REQ-RES-001: atomic publication
# ---------------------------------------------------------------------------


def test_atomic_publish_keeps_previous_version_when_writer_fails(tmp_path: Path) -> None:
    """`TEST-RES-001`."""

    target = tmp_path / "latest.zip"
    target.write_bytes(b"previous-complete")

    def _half_then_raise(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError, match="killed mid-write"):
        atomic_publish(target, _half_then_raise)

    assert target.read_bytes() == b"previous-complete"
    assert not temporary_sibling(target).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest.zip"]


def test_atomic_publish_commits_complete_file_and_keeps_suffix_for_sb3(tmp_path: Path) -> None:
    """`TEST-RES-002`: the temporary name keeps a suffix, so a writer that appends
    ``.zip`` to suffix-less paths (SB3 ``model.save``) leaves it untouched."""

    target = tmp_path / "sub" / "latest.zip"
    seen: list[Path] = []

    def _writer(tmp: Path) -> None:
        seen.append(tmp)
        assert tmp.suffix == ".tmp"
        assert tmp.parent == target.parent
        tmp.write_bytes(b"complete")

    atomic_publish(target, _writer)
    assert target.read_bytes() == b"complete"
    assert not seen[0].exists()


def test_atomic_publish_rejects_writer_that_creates_nothing(tmp_path: Path) -> None:
    target = tmp_path / "state.yaml"
    with pytest.raises(RuntimeError, match="did not create"):
        atomic_publish(target, lambda tmp: None)
    assert not target.exists()


# ---------------------------------------------------------------------------
# Snapshot layout and the `periodic` alias
# ---------------------------------------------------------------------------


def test_resume_artifact_paths_for_latest_and_periodic(tmp_path: Path) -> None:
    latest = resume_artifact_paths(tmp_path, "latest")
    assert latest.model_path == tmp_path / "latest.zip"
    assert latest.training_state_path == tmp_path / "latest_training_state.yaml"
    assert latest.rng_state_path == tmp_path / "latest_rng_state.pkl"
    assert latest.acl_state_dir is None

    final = resume_artifact_paths(tmp_path, "final")
    assert final.model_path == tmp_path / "final.zip"
    assert final.training_state_path == tmp_path / "latest_training_state.yaml"

    periodic = resume_artifact_paths(tmp_path, "periodic/step_00125000")
    assert periodic.is_periodic
    assert periodic.model_path == tmp_path / "periodic" / "step_00125000.zip"
    assert periodic.replay_path == tmp_path / "periodic" / "step_00125000_replay_buffer.pkl"
    assert periodic.pair_path == tmp_path / "periodic" / "step_00125000_checkpoint_pair.json"
    assert (
        periodic.training_state_path == tmp_path / "periodic" / "step_00125000_training_state.yaml"
    )
    assert periodic.acl_state_dir == tmp_path / "periodic" / "step_00125000_acl"


def _make_periodic(checkpoints_dir: Path, step: int, *, paired: bool) -> str:
    name = f"periodic/step_{step:08d}"
    paths = resume_artifact_paths(checkpoints_dir, name)
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    paths.model_path.write_bytes(b"model")
    if paired:
        paths.replay_path.write_bytes(b"replay")
        write_checkpoint_pair(
            paths.pair_path,
            checkpoint_name=name,
            replay_name=paths.replay_path.name,
            training_timestep=step,
            model_num_timesteps=step,
        )
    return name


def test_periodic_alias_resolves_to_newest_complete_pair(tmp_path: Path) -> None:
    _make_periodic(tmp_path, 100_000, paired=True)
    _make_periodic(tmp_path, 200_000, paired=True)
    _make_periodic(tmp_path, 300_000, paired=False)  # model only: not resumable with replay

    assert resolve_resume_checkpoint_name(tmp_path, "periodic") == "periodic/step_00200000"
    assert resolve_resume_checkpoint_name(tmp_path, "latest") == "latest"
    assert (
        resolve_resume_checkpoint_name(tmp_path, "periodic/step_00300000")
        == "periodic/step_00300000"
    )


def test_periodic_alias_fails_with_actionable_message_when_no_pair(tmp_path: Path) -> None:
    _make_periodic(tmp_path, 100_000, paired=False)
    with pytest.raises(FileNotFoundError, match="allow_replay_reset=true"):
        resolve_resume_checkpoint_name(tmp_path, "periodic")


def test_periodic_snapshot_due_uses_crossing_semantics() -> None:
    """`TEST-RES-011`: 125k is never a 50k chunk end, so equality would skip it."""

    assert periodic_snapshot_due(100_000, 150_000, 125_000) is True
    assert periodic_snapshot_due(150_000, 200_000, 125_000) is False
    assert periodic_snapshot_due(200_000, 250_000, 125_000) is True
    assert periodic_snapshot_due(0, 50_000, 50_000) is True
    assert periodic_snapshot_due(50_000, 50_000, 50_000) is False
    assert periodic_snapshot_due(0, 50_000, 0) is False
    assert periodic_snapshot_due(0, 10_000, 10_000) is True


# ---------------------------------------------------------------------------
# REQ-RES-003: torn-snapshot and seed checks
# ---------------------------------------------------------------------------


def test_model_consistency_check_rejects_torn_snapshot(tmp_path: Path) -> None:
    """`TEST-RES-005`."""

    with pytest.raises(ValueError, match="torn"):
        assert_snapshot_model_consistent(
            recorded_model_timesteps=1000,
            loaded_model_timesteps=1500,
            checkpoint_name="latest",
            state_path=tmp_path / "latest_training_state.yaml",
        )


def test_model_consistency_check_accepts_match_and_legacy_snapshot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert_snapshot_model_consistent(
        recorded_model_timesteps=1000,
        loaded_model_timesteps=1000,
        checkpoint_name="latest",
        state_path=tmp_path / "s.yaml",
    )
    logger = logging.getLogger("test_resume_snapshot")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert_snapshot_model_consistent(
            recorded_model_timesteps=None,
            loaded_model_timesteps=1000,
            checkpoint_name="latest",
            state_path=tmp_path / "s.yaml",
            logger=logger,
        )
    assert "torn-snapshot check skipped" in caplog.text


def test_seed_consistency_check() -> None:
    assert_snapshot_seed_consistent(recorded_seed=None, configured_seed=3)
    assert_snapshot_seed_consistent(recorded_seed=3, configured_seed=3)
    with pytest.raises(ValueError, match="seed mismatch"):
        assert_snapshot_seed_consistent(recorded_seed=3, configured_seed=4)


def test_planner_trained_timesteps_prefers_model_counter() -> None:
    planner = SimpleNamespace(model=SimpleNamespace(num_timesteps=1234), num_timesteps=0)
    assert planner_trained_timesteps(planner) == 1234
    assert planner_trained_timesteps(SimpleNamespace(num_timesteps=7)) == 7
    assert planner_trained_timesteps(object()) is None


# ---------------------------------------------------------------------------
# REQ-RES-002 / DEC-RES-004: replay resume classification (REQ-025)
# ---------------------------------------------------------------------------


def _pair(tmp_path: Path, *, with_replay: bool) -> tuple[Path, Path]:
    pair = tmp_path / "latest_checkpoint_pair.json"
    replay = tmp_path / "latest_replay_buffer.pkl"
    write_checkpoint_pair(
        pair, checkpoint_name="latest", replay_name=replay.name, training_timestep=10
    )
    if with_replay:
        replay.write_bytes(b"replay")
    return pair, replay


def test_replay_resume_is_continuation_with_complete_pair(tmp_path: Path) -> None:
    pair, replay = _pair(tmp_path, with_replay=True)
    mode = classify_replay_resume(
        learner_has_replay=True,
        persistence_enabled=True,
        pair_path=pair,
        replay_path=replay,
        allow_replay_reset=False,
        checkpoint_name="latest",
    )
    assert mode == "continuation"


def test_model_only_resume_fails_closed_without_flag(tmp_path: Path) -> None:
    """`TEST-RES-007`."""

    with pytest.raises(ValueError, match="allow_replay_reset=true"):
        classify_replay_resume(
            learner_has_replay=True,
            persistence_enabled=True,
            pair_path=tmp_path / "missing_pair.json",
            replay_path=tmp_path / "missing_replay.pkl",
            allow_replay_reset=False,
            checkpoint_name="latest",
        )
    # Persistence disabled (the pre-ADR-079 production default) is no longer silent.
    with pytest.raises(ValueError, match="persistence disabled"):
        classify_replay_resume(
            learner_has_replay=True,
            persistence_enabled=False,
            pair_path=tmp_path / "missing_pair.json",
            replay_path=tmp_path / "missing_replay.pkl",
            allow_replay_reset=False,
            checkpoint_name="latest",
        )


def test_model_only_resume_with_flag_is_classified_as_reset(tmp_path: Path) -> None:
    """`TEST-RES-008`."""

    mode = classify_replay_resume(
        learner_has_replay=True,
        persistence_enabled=True,
        pair_path=tmp_path / "missing_pair.json",
        replay_path=tmp_path / "missing_replay.pkl",
        allow_replay_reset=True,
        checkpoint_name="latest",
    )
    assert mode == "reset"


def test_partially_committed_pair_is_rejected_even_with_flag(tmp_path: Path) -> None:
    pair, replay = _pair(tmp_path, with_replay=False)
    with pytest.raises(ValueError, match="partially committed"):
        classify_replay_resume(
            learner_has_replay=True,
            persistence_enabled=True,
            pair_path=pair,
            replay_path=replay,
            allow_replay_reset=False,
            checkpoint_name="latest",
        )


def test_on_policy_learner_has_no_replay_classification(tmp_path: Path) -> None:
    mode = classify_replay_resume(
        learner_has_replay=False,
        persistence_enabled=False,
        pair_path=tmp_path / "x.json",
        replay_path=tmp_path / "x.pkl",
        allow_replay_reset=False,
        checkpoint_name="latest",
    )
    assert mode == "not_applicable"


def test_checkpoint_pair_carries_identity_fields(tmp_path: Path) -> None:
    payload = write_checkpoint_pair(
        tmp_path / "periodic" / "step_00000010_checkpoint_pair.json",
        checkpoint_name="periodic/step_00000010",
        replay_name="step_00000010_replay_buffer.pkl",
        training_timestep=10,
        beta_progress_env_steps=7,
        model_num_timesteps=10,
    )
    on_disk = json.loads(
        (tmp_path / "periodic" / "step_00000010_checkpoint_pair.json").read_text(encoding="utf-8")
    )
    assert on_disk == payload
    assert on_disk["model_path"] == "step_00000010.zip"
    assert on_disk["replay_path"] == "step_00000010_replay_buffer.pkl"
    assert on_disk["training_timestep"] == 10
    assert on_disk["beta_progress_env_steps"] == 7
    assert on_disk["model_num_timesteps"] == 10
    assert on_disk["replay_segment_id"] == 0
    assert on_disk["checkpoint_id"]


# ---------------------------------------------------------------------------
# keep_last=1 replay pruning, companion pruning, final cleanup (DEC-RES-006)
# ---------------------------------------------------------------------------


def test_prune_replay_snapshots_keeps_only_the_newest_pair(tmp_path: Path) -> None:
    _make_periodic(tmp_path, 100_000, paired=True)
    newest = _make_periodic(tmp_path, 200_000, paired=True)

    removed = prune_replay_snapshots(tmp_path, keep_checkpoint_name=newest)

    assert sorted(p.name for p in removed) == [
        "step_00100000_checkpoint_pair.json",
        "step_00100000_replay_buffer.pkl",
    ]
    assert (tmp_path / "periodic" / "step_00100000.zip").exists()
    assert (tmp_path / "periodic" / "step_00200000_replay_buffer.pkl").exists()
    assert (tmp_path / "periodic" / "step_00200000_checkpoint_pair.json").exists()


def test_prune_old_periodic_checkpoints_removes_companions_with_the_zip(tmp_path: Path) -> None:
    for step in (100_000, 200_000, 300_000):
        name = _make_periodic(tmp_path, step, paired=False)
        paths = resume_artifact_paths(tmp_path, name)
        atomic_write_text(paths.training_state_path, "global_steps_done: 1\n")
        paths.rng_state_path.write_bytes(b"rng")
        acl_dir = paths.acl_state_dir
        assert acl_dir is not None
        acl_dir.mkdir()
        (acl_dir / "scenario_acl_state.json").write_text("{}", encoding="utf-8")
        (paths.checkpoint_stem.parent / f"{paths.checkpoint_stem.name}.adapter.pt").write_bytes(
            b"a"
        )

    prune_old_periodic_checkpoints(tmp_path, keep_last=2)

    remaining = sorted(p.name for p in (tmp_path / "periodic").iterdir())
    assert not any(name.startswith("step_00100000") for name in remaining)
    assert "step_00200000.zip" in remaining and "step_00300000.zip" in remaining
    assert "step_00300000_training_state.yaml" in remaining
    assert "step_00300000_acl" in remaining


def test_remove_replay_snapshots_after_final_drops_intermediate_copies(tmp_path: Path) -> None:
    _make_periodic(tmp_path, 100_000, paired=True)
    (tmp_path / "latest_replay_buffer.pkl").write_bytes(b"legacy")
    (tmp_path / "latest_checkpoint_pair.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final_replay_buffer.pkl").write_bytes(b"final")
    (tmp_path / "final_checkpoint_pair.json").write_text("{}", encoding="utf-8")

    removed = remove_replay_snapshots_after_final(tmp_path)

    assert len(removed) == 4
    assert (tmp_path / "final_replay_buffer.pkl").exists()
    assert (tmp_path / "final_checkpoint_pair.json").exists()
    assert (tmp_path / "periodic" / "step_00100000.zip").exists()
    assert not (tmp_path / "periodic" / "step_00100000_replay_buffer.pkl").exists()


# ---------------------------------------------------------------------------
# REQ-RES-007: best keys survive the round trip
# ---------------------------------------------------------------------------


def test_best_keys_round_trip_through_training_state(tmp_path: Path) -> None:
    """`TEST-RES-013`."""

    payload = _best_keys_payload(
        (0.1, 0.0, float("inf"), 0.0, -0.9, -0.8, -12.5), None, (1.0, 0.25, 0.0, 0.0)
    )
    state_path = tmp_path / "latest_training_state.yaml"
    OmegaConf.save(config=OmegaConf.create({"best_keys": payload}), f=str(state_path))
    loaded = OmegaConf.to_container(OmegaConf.load(state_path), resolve=True)

    keys = loaded["best_keys"]
    assert _restore_best_key(keys, "lexicographic") == (
        0.1,
        0.0,
        float("inf"),
        0.0,
        -0.9,
        -0.8,
        -12.5,
    )
    assert _restore_best_key(keys, "rulebook_strict") is None
    assert _restore_best_key(keys, "rulebook_thresholded") == (1.0, 0.25, 0.0, 0.0)
    assert _restore_best_key(None, "lexicographic") is None


# ---------------------------------------------------------------------------
# DEC-RES-002: SIGTERM handled like Ctrl+C
# ---------------------------------------------------------------------------


def test_sigterm_is_raised_as_keyboard_interrupt() -> None:
    """`TEST-RES-009` (unit form; the process-level form is the smoke evidence)."""

    previous = signal.getsignal(signal.SIGTERM)
    try:
        assert threading.current_thread() is threading.main_thread()
        assert install_sigterm_as_keyboard_interrupt() is True
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            # Give the interpreter a chance to run the handler.
            signal.pthread_sigmask(signal.SIG_BLOCK, [])
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------
# Scenario ACL: frozen curriculum state per periodic snapshot
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save(self, stem: Path) -> None:
        target = Path(stem).with_suffix(".zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        self.saved.append(target)


class _FakePlanner:
    def __init__(self, num_timesteps: int) -> None:
        self.model = SimpleNamespace(num_timesteps=num_timesteps)

    def save_replay_buffer(self, path: str) -> bool:
        Path(path).write_bytes(b"replay-" + str(self.model.num_timesteps).encode())
        return True


def _acl_fixture(tmp_path: Path):
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    for directory in (artifacts / "curriculum", checkpoints, logs):
        directory.mkdir(parents=True)
    paths = ScenarioAclDriverPaths(
        artifacts_dir=artifacts,
        run_dir=run_dir,
        checkpoints_dir=checkpoints,
        final_checkpoint_stem=checkpoints / "final",
        latest_checkpoint_stem=checkpoints / "latest",
        events_log_path=logs / "events.jsonl",
    )
    artifact_paths = {
        "root": artifacts / "curriculum",
        "state": artifacts / "curriculum" / "scenario_acl_state.json",
        "buffer": artifacts / "curriculum" / "scenario_buffer.json",
        "coverage": artifacts / "curriculum" / "scenario_coverage_state.json",
        "vector_state": artifacts / "curriculum" / "scenario_acl_vector_state.json",
    }
    cfg = OmegaConf.create(
        {
            "seed": 5,
            "checkpoint": {
                "save_latest_each_chunk": True,
                "save_rng_state": False,
                "save_periodic": True,
                "periodic_interval_steps": 100,
                "keep_last_periodic": 4,
            },
        }
    )
    return paths, artifact_paths, cfg


def test_acl_state_dir_mapping(tmp_path: Path) -> None:
    assert (
        _acl_state_dir_for_checkpoint(tmp_path, "latest") == tmp_path / "artifacts" / "curriculum"
    )
    assert _acl_state_dir_for_checkpoint(tmp_path, "final") == tmp_path / "artifacts" / "curriculum"
    assert (
        _acl_state_dir_for_checkpoint(tmp_path, "periodic/step_00000200")
        == tmp_path / "checkpoints" / "periodic" / "step_00000200_acl"
    )


def test_acl_state_files_are_written_with_state_last(tmp_path: Path) -> None:
    root = tmp_path / "curriculum"
    _write_acl_state_files(
        root=root,
        acl_state_payload={"global_step": 3},
        buffer=ScenarioBuffer(capacity=4),
        visit_state=ScenarioCatalogVisitState(["a", "b"]),
        vector_state=None,
    )
    assert json.loads((root / "scenario_acl_state.json").read_text())["global_step"] == 3
    assert (root / "scenario_buffer.json").is_file()
    assert (root / "scenario_coverage_state.json").is_file()
    assert not (root / "scenario_acl_vector_state.json").exists()
    assert not any(p.name.endswith(".tmp") for p in root.iterdir())


def test_acl_resume_snapshot_writes_latest_and_paired_periodic_when_due(tmp_path: Path) -> None:
    paths, artifact_paths, cfg = _acl_fixture(tmp_path)
    agent = _FakeAgent()
    replay_cfg = SimpleNamespace(persistence_enabled=True, periodic_replay_persistence=True)
    logger = logging.getLogger("test_acl_snapshot")
    common = dict(
        cfg=cfg,
        paths=paths,
        agent=agent,
        transition_replay_config=replay_cfg,
        artifact_paths=artifact_paths,
        buffer=ScenarioBuffer(capacity=4),
        visit_state=ScenarioCatalogVisitState(["a", "b"]),
        vector_state=None,
        train_logger=logger,
    )

    # Chunk end at step 50: latest only, no periodic crossing yet.
    last = _write_acl_resume_snapshot(
        planner=_FakePlanner(50),
        acl_state_payload={"global_step": 50},
        current_global_step=50,
        last_snapshot_global_step=0,
        **common,
    )
    assert last == 50
    assert (paths.checkpoints_dir / "latest.zip").is_file()
    live_state = json.loads(artifact_paths["state"].read_text())
    assert live_state["model_num_timesteps"] == 50 and live_state["seed"] == 5
    assert not (paths.checkpoints_dir / "periodic").exists()

    # Chunk end at step 150 crosses 100: periodic snapshot with replay pair + frozen ACL copy.
    last = _write_acl_resume_snapshot(
        planner=_FakePlanner(150),
        acl_state_payload={"global_step": 150},
        current_global_step=150,
        last_snapshot_global_step=last,
        **common,
    )
    periodic = resume_artifact_paths(paths.checkpoints_dir, "periodic/step_00000150")
    assert periodic.model_path.is_file()
    assert periodic.replay_path.read_bytes() == b"replay-150"
    pair = json.loads(periodic.pair_path.read_text())
    assert pair["training_timestep"] == 150 and pair["model_num_timesteps"] == 150
    assert periodic.acl_state_dir is not None
    frozen = json.loads((periodic.acl_state_dir / "scenario_acl_state.json").read_text())
    assert frozen["global_step"] == 150
    assert resolve_resume_checkpoint_name(paths.checkpoints_dir, "periodic") == (
        "periodic/step_00000150"
    )

    # Step 250 crosses 200: the previous replay pair is pruned (keep_last=1), the model kept.
    _write_acl_resume_snapshot(
        planner=_FakePlanner(250),
        acl_state_payload={"global_step": 250},
        current_global_step=250,
        last_snapshot_global_step=last,
        **common,
    )
    assert periodic.model_path.is_file()
    assert not periodic.replay_path.exists()
    assert not periodic.pair_path.exists()
    newest = resume_artifact_paths(paths.checkpoints_dir, "periodic/step_00000250")
    assert newest.replay_path.is_file() and newest.pair_path.is_file()
    assert resolve_resume_checkpoint_name(paths.checkpoints_dir, "periodic") == (
        "periodic/step_00000250"
    )
    events = [json.loads(line) for line in paths.events_log_path.read_text().splitlines()]
    assert [e["event"] for e in events].count("replay_snapshot_written") == 2
