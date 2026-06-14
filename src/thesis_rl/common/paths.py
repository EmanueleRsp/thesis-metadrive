from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "default_analysis_root",
    "default_analysis_root_str",
    "default_output_path_str",
    "default_outputs_glob_str",
    "default_outputs_root",
    "default_outputs_root_str",
]


def _current_user() -> str:
    user = os.environ.get("USER", "").strip()
    return user or "e.respino"


def default_outputs_root() -> Path:
    return Path("/scratch") / _current_user() / "thesis-metadrive" / "outputs"


def default_outputs_root_str() -> str:
    return str(default_outputs_root())


def default_analysis_root() -> Path:
    return default_outputs_root() / "analysis"


def default_analysis_root_str() -> str:
    return str(default_analysis_root())


def default_output_path_str(*parts: str) -> str:
    return str(default_outputs_root().joinpath(*parts))


def default_outputs_glob_str(pattern: str) -> str:
    return str(default_outputs_root() / pattern)
