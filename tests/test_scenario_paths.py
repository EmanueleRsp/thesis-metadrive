from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.scenarios.paths import DATA_DIRECTORIES, ScenarioDataPaths


def test_scenario_data_paths_create_idempotent_layout(tmp_path: Path) -> None:
    paths = ScenarioDataPaths(tmp_path / "scenarionet")

    first = paths.ensure_layout()
    second = paths.ensure_layout()

    assert first == second
    assert len(first) == len(DATA_DIRECTORIES)
    assert all(path.is_dir() for path in first)
    assert paths.runtime_split("validation") == paths.root / "runtime" / "validation"


@pytest.mark.parametrize("relative", ["/tmp/scenario.pkl", "../scenario.pkl", "pg/../x.pkl"])
def test_resolve_relative_rejects_escape(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="unsafe|normalized|escapes"):
        ScenarioDataPaths(tmp_path).resolve_relative(relative)


def test_make_relative_rejects_external_path(tmp_path: Path) -> None:
    paths = ScenarioDataPaths(tmp_path / "root")
    paths.ensure_layout()
    internal = paths.root / "pg" / "database" / "scenario.pkl"

    assert paths.make_relative(internal) == "pg/database/scenario.pkl"
    with pytest.raises(ValueError, match="outside"):
        paths.make_relative(tmp_path / "external.pkl")
