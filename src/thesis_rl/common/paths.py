from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "default_analysis_root",
    "default_analysis_root_str",
    "default_data_root",
    "default_data_root_str",
    "default_metadrive_data_root",
    "default_metadrive_data_root_str",
    "default_output_path_str",
    "default_outputs_glob_str",
    "default_outputs_root",
    "default_outputs_root_str",
    "default_scenarionet_data_root",
    "default_scenarionet_data_root_str",
]


def _first_env_path(*keys: str, default: str) -> Path:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value)
    return Path(default)


def default_outputs_root() -> Path:
    return _first_env_path("OUTPUTS_ROOT", "THESIS_OUTPUT_DIR", default="/workspace/outputs")


def default_outputs_root_str() -> str:
    return str(default_outputs_root())


def default_data_root() -> Path:
    return _first_env_path("DATA_ROOT", "THESIS_DATA_DIR", default="/workspace/data")


def default_data_root_str() -> str:
    return str(default_data_root())


def default_analysis_root() -> Path:
    return default_outputs_root() / "analysis"


def default_analysis_root_str() -> str:
    return str(default_analysis_root())


def default_scenarionet_data_root() -> Path:
    return _first_env_path(
        "SCENARIONET_DATA_ROOT",
        default=str(default_data_root() / "scenarionet"),
    )


def default_scenarionet_data_root_str() -> str:
    return str(default_scenarionet_data_root())


def default_metadrive_data_root() -> Path:
    return _first_env_path(
        "METADRIVE_DATA_ROOT",
        default=str(default_data_root() / "metadrive"),
    )


def default_metadrive_data_root_str() -> str:
    return str(default_metadrive_data_root())


def default_output_path_str(*parts: str) -> str:
    return str(default_outputs_root().joinpath(*parts))


def default_outputs_glob_str(pattern: str) -> str:
    return str(default_outputs_root() / pattern)
