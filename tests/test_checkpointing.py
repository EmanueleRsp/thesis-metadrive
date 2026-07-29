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
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV12
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


def test_checkpoint_manifest_infers_v12_identity_for_semantic_v3() -> None:
    manifest = build_checkpoint_manifest(
        observation_type="semantic_v3",
        flat_dim=3009,
        raw_token_count=143,
        encoder_type="latent_query_v3",
        encoder_config={"output_dim": 256, "depth": 4},
        features_dim=256,
        share_features_extractor=False,
        ppo_ortho_init=None,
        algorithm="td3_sb3",
        sb3_version="test",
        sb3_commit="test",
        git_commit="test",
        seed=7,
    )

    assert manifest.observation_schema_version == "1.2-perception-bounded"
    assert manifest.flat_dim == 3009
    assert manifest.raw_token_count == 143


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


def test_checkpoint_compatibility_rejects_scalarization_identity_change() -> None:
    manifest = _manifest()
    changed = build_checkpoint_manifest(
        observation_type="semantic_v2",
        flat_dim=2541,
        raw_token_count=122,
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
        scalarization_specification_id="SCAL-V1.0",
        scalarization_version="1.0",
        scalarization_mode="bounded_satisfaction_rank",
        scalarization_vector_schema_id="rulebook_v2_macro_v4",
        scalarization_priority_base=2.01,
        scalarization_numerical_tolerance=1.0e-8,
        scalarization_native_environment_reward_weight=0.0,
    )
    with pytest.raises(CheckpointCompatibilityError, match="scalarization_specification_id"):
        from thesis_rl.contracts.checkpoint_manifest import assert_checkpoint_compatible

        assert_checkpoint_compatible(manifest, changed)


def test_semantic_v3_manifest_uses_v12_schema_identity_and_rejects_v2() -> None:
    v3 = build_checkpoint_manifest(
        observation_type="semantic_v3",
        flat_dim=3009,
        raw_token_count=143,
        encoder_type="latent_query_v3",
        encoder_config={"output_dim": 256, "depth": 4},
        encoder_architecture_version="1.1-perception-bounded",
        features_dim=256,
        share_features_extractor=False,
        ppo_ortho_init=None,
        algorithm="td3_sb3",
        sb3_version="2.9.0",
        sb3_commit="sb3-commit",
        git_commit="project-commit",
        seed=42,
    )
    v2 = _manifest()

    assert v3.observation_schema_version == SemanticObservationSchemaV12.version
    assert v3.observation_schema_fingerprint == SemanticObservationSchemaV12().fingerprint_sha256()
    assert v3.raw_token_count == 143
    with pytest.raises(CheckpointCompatibilityError, match="observation_schema_version"):
        from thesis_rl.contracts.checkpoint_manifest import assert_checkpoint_compatible

        assert_checkpoint_compatible(v2, v3)


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
