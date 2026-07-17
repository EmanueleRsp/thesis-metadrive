from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from thesis_rl.scenarios.pg.profiles import PG_PROFILES


_PROFILE_ARM_YIELDS: dict[str, dict[str, float]] = {
    "P0_simple": {"A0_simple_low_traffic": 1.0},
    "P1_vehicle_interaction": {"A0_simple_low_traffic": 1.0},
    "P2_merge_or_roundabout": {
        "A1_traffic": 286 / 350,
        "A2_junction": 52 / 350,
        "A3_complex_junction": 12 / 350,
    },
    "P3_intersection": {
        "A1_traffic": 256 / 350,
        "A2_junction": 89 / 350,
        "A3_complex_junction": 5 / 350,
    },
    "P5_complex_mixed": {
        "A1_traffic": 111 / 350,
        "A2_junction": 122 / 350,
        "A3_complex_junction": 35 / 350,
        "A5_critical_mixed": 82 / 350,
    },
}


def _profile_deficits(report: Mapping[str, Any]) -> dict[str, int]:
    diagnostics = report.get("selection_diagnostics", {})
    global_targets: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        for arm, payload in diagnostics.get(split, {}).get("arms", {}).items():
            global_targets[arm] = global_targets.get(arm, 0) + int(payload.get("target", 0))

    source_arm = report.get("runtime_eligible_by_source_arm")
    if isinstance(source_arm, Mapping):
        pg_available = {str(arm): int(count) for arm, count in source_arm.get("pg", {}).items()}
        waymo_available = {
            str(arm): int(count) for arm, count in source_arm.get("waymo", {}).items()
        }
        return {
            arm: max(
                global_targets.get(arm, 0) - waymo_available.get(arm, 0) - pg_available.get(arm, 0),
                0,
            )
            for arm in global_targets
            if max(
                global_targets.get(arm, 0) - waymo_available.get(arm, 0) - pg_available.get(arm, 0),
                0,
            )
            > 0
        }

    # Compatibility fallback for reports written before the source×arm field.
    available = {
        str(arm): int(count) for arm, count in report.get("runtime_eligible_by_arm", {}).items()
    }
    pg_targets: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        for arm, payload in diagnostics.get(split, {}).get("arms", {}).items():
            source_payload = payload.get("sources", {}).get("pg", {})
            pg_targets[arm] = pg_targets.get(arm, 0) + int(source_payload.get("target", 0))
    return {
        arm: max(pg_targets.get(arm, 0) - available.get(arm, 0), 0)
        for arm in pg_targets
        if max(pg_targets.get(arm, 0) - available.get(arm, 0), 0) > 0
    }


def plan_profile_counts(report: Mapping[str, Any], *, budget: int = 1750) -> dict[str, int]:
    """Allocate a bounded PG candidate budget using the frozen pilot yield matrix."""

    if budget < 1:
        raise ValueError("budget must be positive")
    deficits = _profile_deficits(report)
    if not deficits:
        return {}
    scores = {
        profile.name: sum(
            deficits.get(arm, 0) * yield_rate
            for arm, yield_rate in _PROFILE_ARM_YIELDS[profile.name].items()
        )
        for profile in PG_PROFILES
    }
    useful = {profile: score for profile, score in scores.items() if score > 0}
    if not useful:
        return {}
    total_score = sum(useful.values())
    counts = {profile: int(budget * score / total_score) for profile, score in useful.items()}
    for profile in sorted(useful, key=lambda name: (-useful[name], name)):
        if sum(counts.values()) >= budget:
            break
        counts[profile] += 1
    return {profile: count for profile, count in counts.items() if count > 0}


def load_report(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
