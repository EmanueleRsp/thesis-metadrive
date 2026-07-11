from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from thesis_rl.scenarios.bootstrap import (
    _read_git_commit,
    create_initial_manifest,
    initialize_bootstrap_artifacts,
)


def _inventory() -> dict[str, object]:
    return {
        "generated_at": "2026-07-11T00:00:00+00:00",
        "python": "3.10.20",
        "packages": {
            "metadrive-simulator": "0.4.3",
            "scenarionet": "0.0.1",
        },
        "git": {
            "project": {"commit": "project-commit", "dirty": True},
            "metadrive": {"commit": "metadrive-commit", "dirty": False},
            "scenarionet": {"commit": "scenarionet-commit", "dirty": False},
        },
    }


def test_create_initial_manifest_uses_frozen_local_versions() -> None:
    manifest = create_initial_manifest(_inventory(), created_by_command="bootstrap")

    assert manifest["dataset_id"] == "scenarionet_v1"
    assert manifest["software"]["metadrive_version"] == "0.4.3"
    assert manifest["software"]["scenarionet_commit"] == "scenarionet-commit"
    assert manifest["waymo"]["source_variant"] == "training_20s"
    assert manifest["waymo"]["release"] is None
    assert manifest["scenario_description"]["version"] is None
    assert manifest["creation"]["project_worktree_dirty"] is True


def test_initialize_bootstrap_artifacts_is_non_destructive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "thesis_rl.scenarios.bootstrap.collect_local_api_inventory",
        lambda _repo_root: _inventory(),
    )

    manifest_path, inventory_path = initialize_bootstrap_artifacts(
        data_root=tmp_path,
        repo_root=tmp_path,
        created_by_command="bootstrap",
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert manifest["software"]["project_commit"] == "project-commit"
    assert inventory["packages"]["scenarionet"] == "0.0.1"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        initialize_bootstrap_artifacts(
            data_root=tmp_path,
            repo_root=tmp_path,
            created_by_command="bootstrap",
        )


def test_read_git_commit_supports_submodule_gitdir_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    git_directory = tmp_path / "modules" / "example"
    reference = git_directory / "refs" / "heads" / "main"
    repository.mkdir()
    reference.parent.mkdir(parents=True)
    (repository / ".git").write_text("gitdir: ../modules/example\n", encoding="utf-8")
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reference.write_text("abc123\n", encoding="utf-8")

    assert _read_git_commit(repository) == "abc123"
