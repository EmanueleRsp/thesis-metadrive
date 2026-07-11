from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import pkgutil
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.manifests import validate_dataset_manifest
from thesis_rl.scenarios.paths import ScenarioDataPaths


def _git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _resolve_git_directory(repo: Path) -> Path | None:
    marker = repo / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    content = marker.read_text(encoding="utf-8").strip()
    prefix = "gitdir: "
    if not content.startswith(prefix):
        return None
    return (repo / content[len(prefix) :]).resolve()


def _read_git_commit(repo: Path) -> str | None:
    git_directory = _resolve_git_directory(repo)
    if git_directory is None:
        return None
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    prefix = "ref: "
    if not head.startswith(prefix):
        return head or None
    reference = head[len(prefix) :]
    loose_reference = git_directory / reference
    if loose_reference.is_file():
        return loose_reference.read_text(encoding="utf-8").strip() or None
    packed_refs = git_directory / "packed-refs"
    if packed_refs.is_file():
        suffix = f" {reference}"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("#", "^")) and line.endswith(suffix):
                return line.split(" ", 1)[0]
    return None


def _git_state(repo: Path) -> dict[str, object]:
    commit = _git_output(repo, "rev-parse", "HEAD") or _read_git_commit(repo)
    status = _git_output(repo, "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jsonable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_config_value(item) for key, item in value.items()}
    if inspect.isclass(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _scenario_env_inventory() -> dict[str, object]:
    from metadrive.envs.scenario_env import ScenarioEnv  # type: ignore[import-not-found]

    config = ScenarioEnv.default_config()
    relevant_keys = (
        "data_directory",
        "curriculum_level",
        "horizon",
        "allowed_more_steps",
        "start_scenario_index",
        "num_scenarios",
        "worker_index",
        "num_workers",
        "sequential_seed",
        "agent_policy",
        "set_static",
        "no_map",
        "need_lane_localization",
        "cull_lanes_outside_map",
        "map_region_size",
        "no_traffic",
        "no_static_vehicles",
        "no_light",
        "reactive_traffic",
        "filter_overlapping_car",
        "skip_missing_light",
        "static_traffic_object",
        "store_data",
        "store_map",
        "physics_world_step_size",
        "decision_repeat",
        "crash_vehicle_done",
        "crash_object_done",
        "crash_human_done",
        "out_of_route_done",
        "relax_out_of_road_done",
        "truncate_as_terminate",
    )
    return {
        "class": f"{ScenarioEnv.__module__}.{ScenarioEnv.__qualname__}",
        "reset_signature": str(inspect.signature(ScenarioEnv.reset)),
        "done_function_signature": str(inspect.signature(ScenarioEnv.done_function)),
        "available_config_keys": sorted(str(key) for key in config.keys()),
        "relevant_config": {
            key: _jsonable_config_value(config[key])
            for key in relevant_keys
            if key in config
        },
        "selection_hook_candidates": [
            name
            for name in ("reset", "_reset_global_seed", "seed")
            if hasattr(ScenarioEnv, name)
        ],
        "out_of_road_native_contract": {
            "aggregated_method": "ScenarioEnv._is_out_of_road",
            "continuous_line_flags": [
                "vehicle.on_yellow_continuous_line",
                "vehicle.on_white_continuous_line",
            ],
            "physical_boundary_candidates": [
                "vehicle.crash_sidewalk",
                "not vehicle.on_lane",
            ],
            "route_deviation_in_done_function": "vehicle.navigation.route_completion < -0.1",
            "requires_thesis_override": True,
        },
    }


def _scenario_description_inventory() -> dict[str, object]:
    from metadrive.scenario.scenario_description import (  # type: ignore[import-not-found]
        ScenarioDescription as SD,
    )

    return {
        "class": f"{SD.__module__}.{SD.__qualname__}",
        "first_level_keys": sorted(SD.FIRST_LEVEL_KEYS),
        "metadata_required_keys": sorted(SD.METADATA_KEYS),
        "dataset_summary_file": SD.DATASET.SUMMARY_FILE,
        "dataset_mapping_file": SD.DATASET.MAPPING_FILE,
        "sanity_check_signature": str(inspect.signature(SD.sanity_check)),
    }


def _pg_block_inventory() -> list[dict[str, object]]:
    import metadrive.component.pgblock as pgblock_package  # type: ignore[import-not-found]
    from metadrive.component.pgblock.pg_block import PGBlock  # type: ignore[import-not-found]

    discovered: dict[tuple[str, str], dict[str, object]] = {}
    for module_info in pkgutil.iter_modules(
        pgblock_package.__path__, prefix=f"{pgblock_package.__name__}."
    ):
        if module_info.name.endswith((".create_pg_block_utils", ".pg_block")):
            continue
        module = importlib.import_module(module_info.name)
        for name, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate is PGBlock or not issubclass(candidate, PGBlock):
                continue
            if candidate.__module__ != module.__name__:
                continue
            block_id = candidate.__dict__.get("ID")
            if block_id is None:
                continue
            key = (candidate.__module__, candidate.__qualname__)
            discovered[key] = {
                "class": f"{candidate.__module__}.{candidate.__qualname__}",
                "name": name,
                "id": str(block_id),
                "socket_count": getattr(candidate, "SOCKET_NUM", None),
            }
    return sorted(discovered.values(), key=lambda item: str(item["class"]))


def _scenarionet_commands() -> list[str]:
    import scenarionet  # type: ignore[import-not-found]

    return sorted(
        module.name
        for module in pkgutil.iter_modules(
            scenarionet.__path__, prefix=f"{scenarionet.__name__}."
        )
        if ".tests" not in module.name and ".training" not in module.name
    )


def collect_local_api_inventory(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    metadrive_root = root / "third_party" / "metadrive"
    scenarionet_root = root / "third_party" / "scenarionet"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in (
                "metadrive-simulator",
                "scenarionet",
                "numpy",
                "pandas",
                "pyarrow",
                "geopandas",
            )
        },
        "git": {
            "project": _git_state(root),
            "metadrive": _git_state(metadrive_root),
            "scenarionet": _git_state(scenarionet_root),
        },
        "scenario_env": _scenario_env_inventory(),
        "scenario_description": _scenario_description_inventory(),
        "pg_blocks": _pg_block_inventory(),
        "scenarionet_command_modules": _scenarionet_commands(),
    }


def create_initial_manifest(
    inventory: dict[str, Any],
    *,
    created_by_command: str,
) -> dict[str, object]:
    packages = inventory["packages"]
    git = inventory["git"]
    return {
        "dataset_id": "scenarionet_v1",
        "software": {
            "project_commit": git["project"]["commit"],
            "metadrive_version": packages["metadrive-simulator"],
            "metadrive_commit": git["metadrive"]["commit"],
            "scenarionet_commit": git["scenarionet"]["commit"],
            "python_version": inventory["python"],
        },
        "waymo": {
            "release": None,
            "source_variant": "training_20s",
            "converter_commit": git["scenarionet"]["commit"],
            "source_directory": None,
            "converted_directory": None,
        },
        "procedural": {
            "generator_commit": git["metadrive"]["commit"],
            "exporter_commit": git["metadrive"]["commit"],
            "profiles_version": "pg_profiles_v1",
        },
        # This is the version embedded in each converted ScenarioDescription,
        # not the MetaDrive package version. It is frozen after real conversion.
        "scenario_description": {"version": None},
        "creation": {
            "created_at": inventory["generated_at"],
            "created_by_command": created_by_command,
            "project_worktree_dirty": git["project"]["dirty"],
        },
    }


def _write_new_file(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing bootstrap artifact: {path}")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def initialize_bootstrap_artifacts(
    *,
    data_root: str | Path,
    repo_root: str | Path,
    created_by_command: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    root = Path(data_root).expanduser().resolve()
    ScenarioDataPaths(root).ensure_layout()
    inventory = collect_local_api_inventory(repo_root)
    manifest = validate_dataset_manifest(
        create_initial_manifest(inventory, created_by_command=created_by_command)
    )
    manifest_path = root / "manifest.yaml"
    inventory_path = root / "catalog" / "local_api_inventory.json"
    _write_new_file(
        manifest_path,
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        overwrite=overwrite,
    )
    _write_new_file(
        inventory_path,
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    return manifest_path, inventory_path
