"""Adapters for the official ScenarioNet verifier entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal


OfficialCheck = Literal["existence", "simulation", "overlap"]


def build_official_check_command(
    check: OfficialCheck,
    *,
    database_path: str | Path,
    error_file_path: str | Path | None = None,
    num_workers: int = 8,
    overwrite: bool = False,
    other_database_path: str | Path | None = None,
    show_id: bool = False,
) -> list[str]:
    """Build, without executing, a command from the checked-out ScenarioNet API."""

    database = str(Path(database_path).expanduser().resolve())
    if check in {"existence", "simulation"}:
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        module = f"scenarionet.check_{check}"
        command = [
            sys.executable,
            "-m",
            module,
            "--database_path",
            database,
            "--error_file_path",
            str(Path(error_file_path or Path(database).parent / "validation").resolve()),
            "--num_workers",
            str(num_workers),
        ]
        if overwrite:
            command.append("--overwrite")
        return command

    if other_database_path is None:
        raise ValueError("overlap check requires other_database_path")
    command = [
        sys.executable,
        "-m",
        "scenarionet.check_overlap",
        "--d_1",
        database,
        "--d_2",
        str(Path(other_database_path).expanduser().resolve()),
    ]
    if show_id:
        command.append("--show_id")
    return command


def run_official_check(
    check: OfficialCheck,
    *,
    database_path: str | Path,
    error_file_path: str | Path | None = None,
    num_workers: int = 8,
    overwrite: bool = False,
    other_database_path: str | Path | None = None,
    show_id: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = build_official_check_command(
        check,
        database_path=database_path,
        error_file_path=error_file_path,
        num_workers=num_workers,
        overwrite=overwrite,
        other_database_path=other_database_path,
        show_id=show_id,
    )
    return subprocess.run(command, check=True, text=True, capture_output=True)


__all__ = ["OfficialCheck", "build_official_check_command", "run_official_check"]
