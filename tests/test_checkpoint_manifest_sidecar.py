"""Regression tests for the checkpoint-manifest sidecar wiring.

Covers the operational gap discovered while investigating the sac-0/ppo-0/
td3-0 encoder-incompatibility crash: the manifest compatibility contract
(``CheckpointManifest``, ``build_checkpoint_manifest``) had no production
caller. These tests exercise the additive, fail-open wiring added to
``Agent.save``/``load_planner`` without touching the existing ``.zip``
checkpoint convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.contracts.checkpoint_manifest import (
    CheckpointCompatibilityError,
    build_checkpoint_manifest,
    checkpoint_manifest_sidecar_path,
)
from thesis_rl.runtime.wiring.builders import load_planner
from thesis_rl.runtime.wiring.checkpoint_identity import build_current_checkpoint_manifest


def _manifest(**overrides):
    defaults = dict(
        observation_type="semantic_v3",
        flat_dim=3009,
        raw_token_count=143,
        encoder_type="latent_query_v3",
        encoder_config={
            "type": "latent_query_v3",
            "architecture_version": "1.2-perception-bounded",
            "output_dim": 256,
        },
        encoder_architecture_version="1.2-perception-bounded",
        features_dim=256,
        share_features_extractor=False,
        ppo_ortho_init=None,
        algorithm="td3_sb3",
        sb3_version="2.9.0",
        sb3_commit="sb3-commit",
        git_commit="project-commit",
        seed=42,
    )
    defaults.update(overrides)
    return build_checkpoint_manifest(**defaults)


class _Planner:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"agent-model")


class _Adapter:
    requires_training = False


def _cfg(**overrides):
    base = {
        "seed": 42,
        "device": "cpu",
        "obs": {"type": "semantic_v3"},
        "reward": {"behavior": "monitor_only"},
        "agent": {
            "planner": {
                "algorithm": {"name": "td3_sb3"},
                "encoder": {
                    "type": "latent_query_v3",
                    "architecture_version": "1.2-perception-bounded",
                    "output_dim": 256,
                },
                "decoder": {},
            }
        },
    }
    for dotted_key, value in overrides.items():
        cursor = base
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return OmegaConf.create(base)


def _env(flat_dim: int = 3009):
    return SimpleNamespace(observation_space=SimpleNamespace(shape=(flat_dim,)))


def test_agent_save_writes_manifest_sidecar_when_set(tmp_path: Path) -> None:
    agent = Agent(preprocessor=object(), planner=_Planner(), adapter=_Adapter())
    agent.set_checkpoint_manifest(_manifest())

    checkpoint_path = tmp_path / "ckpt"
    agent.save(checkpoint_path)

    sidecar = checkpoint_manifest_sidecar_path(checkpoint_path)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["encoder_architecture_version"] == "1.2-perception-bounded"


def test_agent_save_skips_manifest_sidecar_when_unset(tmp_path: Path) -> None:
    agent = Agent(preprocessor=object(), planner=_Planner(), adapter=_Adapter())

    checkpoint_path = tmp_path / "ckpt"
    agent.save(checkpoint_path)

    assert not checkpoint_manifest_sidecar_path(checkpoint_path).exists()


def test_load_planner_rejects_shape_incompatible_sidecar_before_backend_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "ckpt.zip"
    checkpoint_path.write_bytes(b"fake-zip")
    stale = _manifest(encoder_architecture_version="1.1-perception-bounded")
    sidecar = checkpoint_manifest_sidecar_path(checkpoint_path)
    sidecar.write_text(json.dumps(stale.to_dict()), encoding="utf-8")

    loader_called = False

    def _fake_backend_loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return object()

    monkeypatch.setattr(
        "thesis_rl.agent.planners.factory.load_planner_backend", _fake_backend_loader
    )

    with pytest.raises(CheckpointCompatibilityError, match="encoder_architecture_version"):
        load_planner(_cfg(), checkpoint_path=str(checkpoint_path), env=_env())
    assert not loader_called


def test_load_planner_ignores_sidecar_mismatch_on_provenance_only_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "ckpt.zip"
    checkpoint_path.write_bytes(b"fake-zip")
    stale = _manifest(git_commit="old-commit", sb3_version="0.0.0", seed=1)
    sidecar = checkpoint_manifest_sidecar_path(checkpoint_path)
    sidecar.write_text(json.dumps(stale.to_dict()), encoding="utf-8")

    loader_called = False

    def _fake_backend_loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return object()

    monkeypatch.setattr(
        "thesis_rl.agent.planners.factory.load_planner_backend", _fake_backend_loader
    )

    load_planner(_cfg(), checkpoint_path=str(checkpoint_path), env=_env())
    assert loader_called


def test_load_planner_skips_manifest_check_when_sidecar_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "ckpt.zip"
    checkpoint_path.write_bytes(b"fake-zip")

    loader_called = False

    def _fake_backend_loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return object()

    monkeypatch.setattr(
        "thesis_rl.agent.planners.factory.load_planner_backend", _fake_backend_loader
    )

    load_planner(_cfg(), checkpoint_path=str(checkpoint_path), env=_env())
    assert loader_called


def test_build_current_checkpoint_manifest_flat_dim_from_env_observation_space() -> None:
    manifest = build_current_checkpoint_manifest(_cfg(**{"obs.type": "lidar_state"}), _env(1234))

    assert manifest.flat_dim == 1234
    assert manifest.raw_token_count is None
