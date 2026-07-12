from __future__ import annotations

import sys

import pytest

from thesis_rl.scenarios.official_checks import build_official_check_command


def test_build_existence_check_uses_checked_out_module(tmp_path) -> None:
    command = build_official_check_command(
        "existence",
        database_path=tmp_path / "database",
        error_file_path=tmp_path / "validation",
        num_workers=1,
        overwrite=True,
    )
    assert command[:3] == [sys.executable, "-m", "scenarionet.check_existence"]
    assert "--database_path" in command
    assert "--overwrite" in command


def test_build_simulation_check_rejects_invalid_worker_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_workers"):
        build_official_check_command(
            "simulation", database_path=tmp_path, num_workers=0
        )


def test_build_overlap_check_requires_second_database(tmp_path) -> None:
    with pytest.raises(ValueError, match="other_database_path"):
        build_official_check_command("overlap", database_path=tmp_path)
