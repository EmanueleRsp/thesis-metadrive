"""Offline driving-mission construction and catalog eligibility."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Callable

from thesis_rl.mission.builder import build_driving_mission_from_source
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.parallel import ordered_process_map


MISSION_ELIGIBILITY_SCHEMA = "driving-mission-catalog-eligibility-v1"
MISSION_BUILDER_IDENTITY = "mission-builder-v1.1.1-anchor-builder"
MissionProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class DrivingMissionEligibility:
    """Result of building one immutable mission before split selection."""

    scenario_uid: str
    source: str
    relative_path: str
    eligible: bool
    mission: dict | None
    validation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_uid": self.scenario_uid,
            "source": self.source,
            "relative_path": self.relative_path,
            "eligible": self.eligible,
            "mission_schema": None if self.mission is None else self.mission.get("schema_version"),
            "mission_hash": None if self.mission is None else self.mission.get("mission_hash"),
            "validation_errors": list(self.validation_errors),
        }


def evaluate_driving_mission_entry(
    entry: ScenarioCatalogEntry,
    *,
    data_root: str | Path,
) -> DrivingMissionEligibility:
    """Build one mission from the frozen source route and fail closed."""

    record = entry.record
    source_path = Path(data_root).expanduser().resolve() / record.relative_path
    identity = {
        "scenario_uid": record.scenario_uid,
        "source": record.source,
        "relative_path": record.relative_path,
    }
    try:
        if not record.assigned_route_lane_ids:
            raise ValueError("assigned_route_lane_ids are missing")
        with source_path.open("rb") as handle:
            scenario = pickle.load(handle)
        mission = build_driving_mission_from_source(
            scenario,
            scenario_uid=record.scenario_uid,
            source=record.source,
            assigned_route_lane_ids=record.assigned_route_lane_ids,
        )
        payload = mission.to_dict()
        return DrivingMissionEligibility(
            **identity,
            eligible=True,
            mission=payload,
            validation_errors=(),
        )
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
        return DrivingMissionEligibility(
            **identity,
            eligible=False,
            mission=None,
            validation_errors=(f"mission_source_load_error:{type(error).__name__}",),
        )
    except Exception as error:
        return DrivingMissionEligibility(
            **identity,
            eligible=False,
            mission=None,
            validation_errors=(f"mission_build_error:{type(error).__name__}:{error}",),
        )


def evaluate_driving_mission_entries(
    entries: tuple[ScenarioCatalogEntry, ...],
    *,
    data_root: str | Path,
    workers: int = 1,
    progress_callback: MissionProgressCallback | None = None,
) -> tuple[DrivingMissionEligibility, ...]:
    """Build missions in deterministic catalog order using bounded workers."""

    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = tuple(sorted(entries, key=lambda entry: entry.record.scenario_uid))
    tasks = tuple((entry, str(data_root)) for entry in ordered)
    return ordered_process_map(
        tasks,
        _evaluate_driving_mission_entry_task,
        workers=workers,
        progress_callback=progress_callback,
    )


def _evaluate_driving_mission_entry_task(
    task: tuple[ScenarioCatalogEntry, str],
) -> DrivingMissionEligibility:
    entry, data_root = task
    return evaluate_driving_mission_entry(entry, data_root=data_root)


__all__ = [
    "MISSION_BUILDER_IDENTITY",
    "MISSION_ELIGIBILITY_SCHEMA",
    "DrivingMissionEligibility",
    "evaluate_driving_mission_entries",
    "evaluate_driving_mission_entry",
]
