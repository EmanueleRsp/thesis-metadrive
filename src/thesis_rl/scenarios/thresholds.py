from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry


@dataclass(frozen=True, slots=True)
class ArmThresholds:
    feature_version: str
    relevant_radius_m: float
    vertical_tolerance_m: float
    temporal_quantile: float
    low_traffic_quantile: float
    dense_traffic_quantile: float
    tau_low: int
    tau_dense: int
    computed_on_split: str = "train"
    balanced_sources: bool = True
    balanced_source_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_arm_thresholds(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    feature_version: str = "v1",
    relevant_radius_m: float = 50.0,
    vertical_tolerance_m: float = 3.0,
    temporal_quantile: float = 0.90,
    low_traffic_quantile: float = 0.40,
    dense_traffic_quantile: float = 0.75,
    balance_seed: int = 0,
) -> ArmThresholds:
    if not entries:
        raise ValueError("threshold computation requires train entries")
    non_train = [entry.record.scenario_uid for entry in entries if entry.record.split != "train"]
    if non_train:
        raise ValueError(f"threshold computation is train-only; received: {non_train}")
    source_entries: dict[str, list[ScenarioCatalogEntry]] = {"waymo": [], "pg": []}
    for entry in entries:
        source_entries[entry.record.source].append(entry)
    counts = {source: len(source_entries[source]) for source in source_entries}
    if not counts["waymo"] or not counts["pg"]:
        raise ValueError(f"threshold computation requires non-empty sources: {counts}")

    # Whole-group splitting is intentionally allowed to overshoot a target. Use
    # a deterministic balanced train subset for threshold estimation instead of
    # failing when, for example, Waymo has 1054 entries and PG has 1000.
    balanced_count = min(counts.values())
    source_values: dict[str, list[float]] = {}
    for source_index, source in enumerate(("waymo", "pg")):
        candidates = sorted(
            source_entries[source],
            key=lambda entry: entry.record.scenario_uid,
        )
        if len(candidates) > balanced_count:
            rng = np.random.default_rng(int(balance_seed) + source_index)
            selected = np.sort(
                rng.choice(len(candidates), size=balanced_count, replace=False)
            )
            candidates = [candidates[int(index)] for index in selected]
        source_values[source] = [
            entry.features.relevant_agents_q90 for entry in candidates
        ]
    values = np.asarray(source_values["waymo"] + source_values["pg"], dtype=np.float64)
    tau_low = int(np.rint(np.quantile(values, low_traffic_quantile)))
    tau_dense = int(np.rint(np.quantile(values, dense_traffic_quantile)))
    return ArmThresholds(
        feature_version=feature_version,
        relevant_radius_m=relevant_radius_m,
        vertical_tolerance_m=vertical_tolerance_m,
        temporal_quantile=temporal_quantile,
        low_traffic_quantile=low_traffic_quantile,
        dense_traffic_quantile=dense_traffic_quantile,
        tau_low=tau_low,
        tau_dense=tau_dense,
        balanced_source_count=balanced_count,
    )


def apply_traffic_thresholds(
    entry: ScenarioCatalogEntry, thresholds: ArmThresholds
) -> ScenarioCatalogEntry:
    value = entry.features.relevant_agents_q90
    return ScenarioCatalogEntry(
        record=entry.record,
        features=replace(
            entry.features,
            low_traffic=value <= thresholds.tau_low,
            dense_traffic=value >= thresholds.tau_dense,
        ),
    )


def write_arm_thresholds(
    thresholds: ArmThresholds, path: str | Path, *, overwrite: bool = False
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite arm thresholds: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(thresholds.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_arm_thresholds(path: str | Path) -> ArmThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = ArmThresholds(**payload)
    if thresholds.computed_on_split != "train" or not thresholds.balanced_sources:
        raise ValueError("arm thresholds must be computed on balanced train sources")
    return thresholds
