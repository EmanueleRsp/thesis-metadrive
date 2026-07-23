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
