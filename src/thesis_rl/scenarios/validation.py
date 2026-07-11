from __future__ import annotations

import pickle
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from thesis_rl.scenarios.features import extract_scenario_features
from thesis_rl.scenarios.records import ScenarioRecord


@dataclass(frozen=True, slots=True)
class ScenarioValidationResult:
    scenario_uid: str
    status: str
    warnings: tuple[str, ...]
    scenario_length: int | None
    reset_smoke_passed: bool | None = None


def validate_scenario_file(
    path: str | Path,
    record: ScenarioRecord,
    *,
    run_feature_extraction: bool = True,
) -> ScenarioValidationResult:
    scenario_path = Path(path)
    warnings: list[str] = []
    try:
        with scenario_path.open("rb") as handle:
            scenario = pickle.load(handle)
        from metadrive.scenario.scenario_description import (  # type: ignore[import-not-found]
            ScenarioDescription as SD,
        )

        SD.sanity_check(scenario, check_self_type=True)
        if run_feature_extraction:
            extract_scenario_features(scenario, record.source)
        essential = ("tracks", "map_features", "metadata", "length")
        missing = [key for key in essential if key not in scenario]
        if missing:
            warnings.append(f"missing essential keys: {missing}")
        length = int(scenario.get("length", 0))
        if length != record.length:
            warnings.append(f"catalog length {record.length} != file length {length}")
        if not np.isfinite(float(length)) or length <= 0:
            warnings.append("scenario length is invalid")
    except Exception as exc:
        return ScenarioValidationResult(
            scenario_uid=record.scenario_uid,
            status="invalid",
            warnings=(f"{type(exc).__name__}: {exc}",),
            scenario_length=None,
        )
    return ScenarioValidationResult(
        scenario_uid=record.scenario_uid,
        status="valid" if not warnings else "warning",
        warnings=tuple(warnings),
        scenario_length=length,
    )


def validate_records(
    records: Sequence[ScenarioRecord],
    *,
    data_root: str | Path,
) -> tuple[ScenarioValidationResult, ...]:
    root = Path(data_root).expanduser().resolve()
    results: list[ScenarioValidationResult] = []
    for record in records:
        path = root / Path(record.relative_path)
        results.append(validate_scenario_file(path, record))
    return tuple(results)


def validation_summary(
    results: Sequence[ScenarioValidationResult], *, catalog_hash: str | None = None
) -> dict[str, object]:
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("valid", "warning", "invalid")
    }
    return {
        "total": len(results),
        "counts": counts,
        "catalog_hash": catalog_hash,
        "invalid_scenario_uids": [
            result.scenario_uid for result in results if result.status == "invalid"
        ],
        "warnings": {
            result.scenario_uid: list(result.warnings)
            for result in results
            if result.warnings
        },
    }


def write_validation_summary(
    results: Sequence[ScenarioValidationResult],
    path: str | Path,
    *,
    catalog_hash: str | None = None,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite validation summary: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(validation_summary(results, catalog_hash=catalog_hash), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
