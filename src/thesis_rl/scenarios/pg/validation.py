from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PGValidationResult:
    status: str
    warnings: tuple[str, ...]
    scenario_length: int
    scenario_id: str


def validate_exported_scenario(scenario: dict[str, Any]) -> PGValidationResult:
    warnings: list[str] = []
    try:
        from metadrive.scenario.scenario_description import (  # type: ignore[import-not-found]
            ScenarioDescription as SD,
        )

        SD.sanity_check(scenario, check_self_type=True)
    except Exception as exc:
        return PGValidationResult(
            status="invalid",
            warnings=(f"ScenarioDescription sanity_check failed: {type(exc).__name__}: {exc}",),
            scenario_length=int(scenario.get("length", 0) or 0),
            scenario_id=str(scenario.get("id", "")),
        )

    length = int(scenario["length"])
    scenario_id = str(scenario.get("id", ""))
    metadata = scenario.get("metadata", {})
    tracks = scenario.get("tracks", {})
    map_features = scenario.get("map_features", {})
    if not scenario_id:
        warnings.append("scenario id is empty")
    if not isinstance(metadata, dict) or not metadata.get("sdc_id"):
        warnings.append("SDC metadata is missing")
    if not tracks:
        warnings.append("scenario has no tracks")
    if not map_features:
        warnings.append("scenario has no map features")

    finite_essential = True
    for track in tracks.values() if isinstance(tracks, dict) else ():
        state = track.get("state", {}) if isinstance(track, dict) else {}
        for key in ("position", "heading"):
            if key in state and not np.isfinite(np.asarray(state[key], dtype=np.float64)).all():
                finite_essential = False
    if not finite_essential:
        warnings.append("essential track state contains NaN/Inf")
    return PGValidationResult(
        status="valid" if not warnings else "warning",
        warnings=tuple(warnings),
        scenario_length=length,
        scenario_id=scenario_id,
    )
