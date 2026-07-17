"""Tests for immutable checkpoint generations and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_rl.agent.agent import Agent
from thesis_rl.contracts.checkpoint_manifest import (
    CheckpointCompatibilityError,
    build_checkpoint_manifest,
)
from thesis_rl.sb3_extensions.checkpointing import (
    LATEST_FILENAME,
    load_checkpoint_generation,
    publish_checkpoint_generation,
    resolve_latest_generation,
)


def _manifest(*, observation_type: str = "semantic_v2", flat_dim: int = 2541):
    return build_checkpoint_manifest(
        observation_type=observation_type,
        flat_dim=flat_dim,
        raw_token_count=122 if observation_type == "semantic_v2" else None,
        encoder_type="latent_query_v2",
        encoder_config={"output_dim": 256, "depth": 4},
        features_dim=256,
        share_features_extractor=False,
        ppo_ortho_init=None,
        algorithm="td3_sb3",
        sb3_version="2.9.0",
        sb3_commit="sb3-commit",
        git_commit="project-commit",
        seed=42,
    )


def _save_model(path: Path) -> None:
    path.write_bytes(b"deterministic-model")


def test_checkpoint_generation_round_trip_and_latest_pointer(tmp_path: Path) -> None:
    manifest = _manifest()
    generation = publish_checkpoint_generation(tmp_path, "latest", _save_model, manifest)

    assert generation.model_path.read_bytes() == b"deterministic-model"
    assert generation.manifest_path.exists()
    pointer = json.loads((tmp_path / LATEST_FILENAME).read_text(encoding="utf-8"))
    assert pointer["generation_dir"] == generation.generation_dir.name
    assert pointer["generation_id"] == generation.generation_id

    loaded, resolved = load_checkpoint_generation(
        tmp_path,
        manifest,
        lambda path: path.read_bytes(),
    )
    assert loaded == b"deterministic-model"
    assert resolved.generation_dir == generation.generation_dir


def test_checkpoint_compatibility_rejects_legacy_shape_before_loader(tmp_path: Path) -> None:
    manifest = _manifest()
    publish_checkpoint_generation(tmp_path, "latest", _save_model, manifest)
    incompatible = _manifest(flat_dim=2363)
    loader_called = False

    def loader(_path: Path):
        nonlocal loader_called
        loader_called = True
        return object()

    with pytest.raises(CheckpointCompatibilityError, match="flat_dim"):
        load_checkpoint_generation(tmp_path, incompatible, loader)
    assert not loader_called


def test_stale_or_torn_latest_pointer_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    publish_checkpoint_generation(tmp_path, "latest", _save_model, manifest)
    pointer_path = tmp_path / LATEST_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["model_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(CheckpointCompatibilityError, match="digest mismatch"):
        resolve_latest_generation(tmp_path, manifest)


def test_incomplete_generation_is_rejected_before_loader(tmp_path: Path) -> None:
    manifest = _manifest()
    incomplete = tmp_path / "latest-incomplete"
    incomplete.mkdir()
    (incomplete / "model.zip").write_bytes(b"model")

    with pytest.raises(CheckpointCompatibilityError, match="incomplete"):
        load_checkpoint_generation(
            tmp_path,
            manifest,
            lambda path: path.read_bytes(),
            generation_dir=incomplete,
        )


def test_agent_explicit_generation_save_uses_planner_model_path(tmp_path: Path) -> None:
    class _Planner:
        def save(self, path: Path) -> None:
            path.write_bytes(b"agent-model")

    class _Adapter:
        requires_training = False

    agent = Agent(preprocessor=object(), planner=_Planner(), adapter=_Adapter())
    generation = agent.save_generation(tmp_path, "final", _manifest())

    assert generation.model_path.read_bytes() == b"agent-model"
    assert generation.manifest_path.exists()
