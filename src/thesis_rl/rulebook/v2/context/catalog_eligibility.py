"""Offline Rulebook v2 eligibility for ScenarioNet catalog entries."""

from __future__ import annotations

from dataclasses import replace
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

from thesis_rl.rulebook.v2.config import RULEBOOK_V2_VERSION
from thesis_rl.rulebook.v2.context.map_matching import TaskRouteMapMatchError
from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.task_route import TaskRouteEligibility, validate_task_route
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.parallel import ordered_process_map


CatalogProgressCallback = Callable[[int, int], None]


def _adapter_version(source: str) -> str:
    if source == "pg":
        return "pg-v2"
    if source == "waymo":
        return "waymo-v2"
    raise ValueError(f"Unsupported Rulebook catalog source: {source!r}")


def _excluded(
    *,
    entry: ScenarioCatalogEntry,
    geometry_config_hash: str,
    calibration_hash: str,
    errors: tuple[str, ...],
) -> TaskRouteEligibility:
    return TaskRouteEligibility(
        scenario_uid=entry.record.scenario_uid,
        rulebook_version=RULEBOOK_V2_VERSION,
        adapter_version=_adapter_version(entry.record.source),
        geometry_config_hash=geometry_config_hash,
        calibration_hash=calibration_hash,
        rulebook_eligible=False,
        validation_errors=errors,
    )


def evaluate_catalog_entry(
    entry: ScenarioCatalogEntry,
    *,
    data_root: str | Path,
    geometry_config_hash: str,
    calibration_hash: str,
) -> TaskRouteEligibility:
    """Validate one static scenario and fail closed with a typed audit result."""

    if not geometry_config_hash or not calibration_hash:
        raise ValueError("Rulebook eligibility requires non-empty geometry and calibration hashes")
    path = Path(data_root).expanduser().resolve() / entry.record.relative_path
    try:
        with path.open("rb") as handle:
            scenario: Any = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
        return _excluded(
            entry=entry,
            geometry_config_hash=geometry_config_hash,
            calibration_hash=calibration_hash,
            errors=(f"scenario_load_error:{type(error).__name__}",),
        )

    try:
        if entry.record.source == "pg":
            result = build_pg_static_adapter_result(
                scenario,
                scenario_uid=entry.record.scenario_uid,
            )
        elif entry.record.source == "waymo":
            result = build_waymo_static_adapter_result(
                scenario,
                scenario_uid=entry.record.scenario_uid,
            )
        else:  # guarded by ScenarioRecord, retained for a future source extension
            raise ValueError(f"Unsupported Rulebook catalog source: {entry.record.source!r}")
    except TaskRouteMapMatchError as error:
        return _excluded(
            entry=entry,
            geometry_config_hash=geometry_config_hash,
            calibration_hash=calibration_hash,
            errors=(error.validation_error,),
        )
    except Exception as error:
        return _excluded(
            entry=entry,
            geometry_config_hash=geometry_config_hash,
            calibration_hash=calibration_hash,
            errors=(f"adapter_exception:{type(error).__name__}",),
        )

    route_eligibility = validate_task_route(
        result.task_route,
        available_lane_ids={lane.lane_id: lane for lane in result.route_lanes},
        rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash=geometry_config_hash,
        calibration_hash=calibration_hash,
    )
    errors = tuple((*result.validation_errors, *route_eligibility.validation_errors))
    return replace(
        route_eligibility,
        rulebook_eligible=not errors,
        validation_errors=errors,
    )


def evaluate_catalog_entries(
    entries: Iterable[ScenarioCatalogEntry],
    *,
    data_root: str | Path,
    geometry_config_hash: str,
    calibration_hash: str,
    workers: int = 1,
    progress_callback: CatalogProgressCallback | None = None,
) -> tuple[TaskRouteEligibility, ...]:
    """Evaluate a deterministic catalog order and retain all audit records."""

    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = sorted(entries, key=lambda entry: entry.record.scenario_uid)
    tasks = tuple(
        (entry, str(data_root), geometry_config_hash, calibration_hash) for entry in ordered
    )
    return ordered_process_map(
        tasks,
        _evaluate_catalog_entry_task,
        workers=workers,
        progress_callback=progress_callback,
    )


def _evaluate_catalog_entry_task(
    task: tuple[ScenarioCatalogEntry, str, str, str],
) -> TaskRouteEligibility:
    """Evaluate one catalog entry in a spawned worker process."""

    entry, data_root, geometry_config_hash, calibration_hash = task
    return evaluate_catalog_entry(
        entry,
        data_root=data_root,
        geometry_config_hash=geometry_config_hash,
        calibration_hash=calibration_hash,
    )


__all__ = [
    "CatalogProgressCallback",
    "evaluate_catalog_entries",
    "evaluate_catalog_entry",
]
