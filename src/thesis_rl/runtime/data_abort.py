"""Run-local state and forensic artifacts for typed runtime scenario aborts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATA_ABORT_STATE_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class RuntimeScenarioQuarantine:
    """Serializable run-owned quarantine; never a dataset mutation."""

    scenario_uids: set[str] = field(default_factory=set)
    reason_counts: dict[str, int] = field(default_factory=dict)
    version: int = DATA_ABORT_STATE_VERSION

    def add(self, scenario_uid: str, reason_code: str) -> bool:
        uid = str(scenario_uid).strip()
        if not uid:
            raise ValueError("Runtime scenario quarantine requires a non-empty scenario UID.")
        self.reason_counts[str(reason_code)] = self.reason_counts.get(str(reason_code), 0) + 1
        before = len(self.scenario_uids)
        self.scenario_uids.add(uid)
        return len(self.scenario_uids) != before

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scenario_uids": sorted(self.scenario_uids),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeScenarioQuarantine":
        if int(payload.get("version", -1)) != DATA_ABORT_STATE_VERSION:
            raise ValueError("Runtime scenario quarantine state version is incompatible.")
        return cls(
            scenario_uids={str(value) for value in payload.get("scenario_uids", ())},
            reason_counts={str(key): int(value) for key, value in dict(payload.get("reason_counts", {})).items()},
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeScenarioQuarantine":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def append_data_abort_record(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append a compact forensic JSONL record and return its JSON-safe content."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_json_safe(dict(payload)),
    }
    final_observation = record.pop("final_observation", None)
    if final_observation is not None:
        encoded = json.dumps(final_observation, sort_keys=True).encode("utf-8")
        record["last_valid_observation_sha256"] = hashlib.sha256(encoded).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")
    return record


GEOMETRY_ABORT_STATE_VERSION = 1

# `DEC-GA-001` and `DEC-GA-002`, approved by the user 2026-09-06. Engineering
# judgement anchored on a single observed occurrence (`open_items` `C9`, roughly
# one in eight evaluation batches), not calibrated constants. Both leave about
# two orders of magnitude of headroom over what was observed.
GEOMETRY_ABORT_MAX_CONSECUTIVE_IN_BATCH = 3
GEOMETRY_ABORT_MAX_RUN_RATE = 0.01
GEOMETRY_ABORT_QUARANTINE_AFTER_REPEATS = 3


@dataclass
class GeometryAbortLedger:
    """Run-local accounting for geometry aborts, with the ceiling that bounds them.

    A geometry abort ends an episode rather than the run. Without a ceiling that
    would be silent absorption with extra steps: a systemic geometry defect would
    show up as a slowly rising discard rate on a field nobody reads, which is the
    outcome `open_items` `C9`'s crash accidentally prevented. The ceiling restores
    the property that a real defect eventually stops the run -- just not on its
    first occurrence, and with the offending geometry already on disk.

    Scenario quarantine is deliberately not immediate. Unlike an `RSA-V1` data
    abort, whether a geometry failure fires depends on where the actors are and
    therefore on the policy, so one occurrence is not evidence the record is
    broken. Repeats on the same UID are.
    """

    max_consecutive_in_batch: int = GEOMETRY_ABORT_MAX_CONSECUTIVE_IN_BATCH
    max_run_rate: float = GEOMETRY_ABORT_MAX_RUN_RATE
    quarantine_after_repeats: int = GEOMETRY_ABORT_QUARANTINE_AFTER_REPEATS

    total: int = 0
    episodes_attempted: int = 0
    consecutive_in_batch: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    per_uid_counts: dict[str, int] = field(default_factory=dict)
    quarantined_uids: set[str] = field(default_factory=set)
    version: int = GEOMETRY_ABORT_STATE_VERSION

    def begin_batch(self) -> None:
        """Reset the within-batch run of consecutive aborts."""

        self.consecutive_in_batch = 0

    def note_episode_completed(self) -> None:
        """Count an episode that finished without a geometry abort."""

        self.episodes_attempted += 1
        self.consecutive_in_batch = 0

    def record(self, scenario_uid: str, reason_code: str) -> bool:
        """Record one geometry abort. Returns whether the UID is now quarantined."""

        uid = str(scenario_uid).strip()
        if not uid:
            raise ValueError("A geometry abort requires a non-empty scenario UID.")
        self.total += 1
        self.episodes_attempted += 1
        self.consecutive_in_batch += 1
        self.reason_counts[str(reason_code)] = self.reason_counts.get(str(reason_code), 0) + 1
        self.per_uid_counts[uid] = self.per_uid_counts.get(uid, 0) + 1
        if self.per_uid_counts[uid] >= self.quarantine_after_repeats:
            newly = uid not in self.quarantined_uids
            self.quarantined_uids.add(uid)
            return newly
        return False

    @property
    def run_rate(self) -> float:
        if self.episodes_attempted <= 0:
            return 0.0
        return self.total / self.episodes_attempted

    def ceiling_breach(self) -> str | None:
        """Return the failure message if either ceiling is exceeded, else ``None``."""

        dominant = (
            max(self.reason_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if self.reason_counts
            else "UNKNOWN"
        )
        if self.consecutive_in_batch >= self.max_consecutive_in_batch:
            return (
                f"Geometry aborts reached {self.consecutive_in_batch} consecutively within one "
                f"evaluation batch, at or above the ceiling of {self.max_consecutive_in_batch}. "
                f"Dominant reason: {dominant}. This is a systemic geometry failure, not an "
                "isolated one; see the geometry-abort forensic record for the offending polygons."
            )
        # The rate ceiling is only meaningful once enough episodes exist for a
        # rate to mean anything; below that the consecutive ceiling is the guard.
        minimum_episodes = int(1.0 / self.max_run_rate) if self.max_run_rate > 0.0 else 0
        if self.episodes_attempted >= minimum_episodes and self.run_rate > self.max_run_rate:
            return (
                f"Geometry aborts reached {self.total} of {self.episodes_attempted} episodes "
                f"({self.run_rate:.3%}), above the ceiling of {self.max_run_rate:.3%}. "
                f"Dominant reason: {dominant}. See the geometry-abort forensic record."
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "total": self.total,
            "episodes_attempted": self.episodes_attempted,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "per_uid_counts": dict(sorted(self.per_uid_counts.items())),
            "quarantined_uids": sorted(self.quarantined_uids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryAbortLedger":
        if int(payload.get("version", -1)) != GEOMETRY_ABORT_STATE_VERSION:
            raise ValueError("Geometry abort ledger state version is incompatible.")
        return cls(
            total=int(payload.get("total", 0)),
            episodes_attempted=int(payload.get("episodes_attempted", 0)),
            reason_counts={
                str(key): int(value)
                for key, value in dict(payload.get("reason_counts", {})).items()
            },
            per_uid_counts={
                str(key): int(value)
                for key, value in dict(payload.get("per_uid_counts", {})).items()
            },
            quarantined_uids={str(value) for value in payload.get("quarantined_uids", ())},
        )


def append_geometry_abort_record(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append a geometry-abort forensic record, kept in its own file.

    Separate from the data-abort log by `REQ-GA-004`: the two are different
    conditions with different remedies, and folding them together would let a
    geometry defect hide inside a data-abort rate that is expected to be non-zero.
    The record keeps the offending polygon's WKT so the failure can be rebuilt as
    a test fixture, which is exactly how `C9` was diagnosed.
    """

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_json_safe(dict(payload)),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")
    return record
