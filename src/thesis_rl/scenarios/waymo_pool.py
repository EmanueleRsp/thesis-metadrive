"""Eligibility and coverage accounting for the converted Waymo pool."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry

WAYMO_POOL_POLICY_VERSION = "waymo_pool_v2"


@dataclass(frozen=True)
class WaymoPoolStatus:
    total: int
    eligible: int
    required: int
    by_signal_reliability: dict[str, int]
    eligible_by_arm: dict[str, int]
    source_shards: tuple[str, ...]

    @property
    def deficit(self) -> int:
        return max(0, self.required - self.eligible)

    @property
    def complete(self) -> bool:
        return self.deficit == 0


def summarize_waymo_pool(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    allowed_signal_reliabilities: Sequence[str],
    required: int,
) -> WaymoPoolStatus:
    """Summarize candidates without assigning train/validation/test splits."""

    if required < 0:
        raise ValueError("required Waymo count must be non-negative")
    allowed = frozenset(str(value) for value in allowed_signal_reliabilities)
    if not allowed:
        raise ValueError("allowed signal reliabilities must not be empty")
    uids = [entry.record.scenario_uid for entry in entries]
    if len(uids) != len(set(uids)):
        duplicates = sorted(uid for uid, count in Counter(uids).items() if count > 1)
        raise ValueError(f"converted Waymo pool contains duplicate scenarios: {duplicates[:5]}")

    eligible_entries = [
        entry for entry in entries if entry.features.signal_reliability in allowed
    ]
    signal_counts = Counter(entry.features.signal_reliability for entry in entries)
    arm_counts = Counter(entry.record.primary_arm for entry in eligible_entries)
    source_shards = sorted(
        {
            str(entry.record.source_log_id)
            for entry in entries
            if entry.record.source_log_id
            and str(entry.record.source_log_id).startswith("training_20s.tfrecord-")
        }
    )
    return WaymoPoolStatus(
        total=len(entries),
        eligible=len(eligible_entries),
        required=int(required),
        by_signal_reliability=dict(sorted(signal_counts.items())),
        eligible_by_arm={arm: int(arm_counts[arm]) for arm in ARMS},
        source_shards=tuple(source_shards),
    )


def fingerprint_waymo_database(database_path: str | Path) -> str:
    """Fingerprint candidate files cheaply enough for safe status-cache reuse."""

    root = Path(database_path).expanduser().resolve()
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("sd_*.pkl")):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "WAYMO_POOL_POLICY_VERSION",
    "WaymoPoolStatus",
    "fingerprint_waymo_database",
    "summarize_waymo_pool",
]
