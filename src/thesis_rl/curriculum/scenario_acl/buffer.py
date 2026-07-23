from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from thesis_rl.curriculum.config import ScenarioAclReplaySamplingConfig
from thesis_rl.curriculum.scenario_acl.ranking import assign_ranks, compute_replay_probabilities
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord


@dataclass(frozen=True)
class ScenarioReplaySelection:
    index: int
    record: ScenarioRecord
    probabilities: list[float]


class ScenarioBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ScenarioBuffer capacity must be > 0.")
        self._capacity = int(capacity)
        self._records: list[ScenarioRecord] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[ScenarioRecord]:
        return list(self._records)

    def _refresh_ranks(self) -> None:
        self._records = assign_ranks(self._records)

    def contains_hash(self, scenario_hash: str) -> bool:
        return any(record.scenario_description_hash == scenario_hash for record in self._records)

    def insert(self, record: ScenarioRecord) -> bool:
        if self.contains_hash(record.scenario_description_hash):
            return False
        if len(self._records) < self._capacity:
            self._records.append(record)
            self._refresh_ranks()
            return True

        worst = min(self._records, key=lambda current: current.usefulness)
        if record.usefulness <= worst.usefulness:
            return False

        worst_index = self._records.index(worst)
        self._records[worst_index] = record
        self._refresh_ranks()
        return True

    def update(self, record: ScenarioRecord) -> None:
        for idx, current in enumerate(self._records):
            if current.scenario_id == record.scenario_id:
                self._records[idx] = record
                self._refresh_ranks()
                return
        raise KeyError(f"Scenario record '{record.scenario_id}' not found in buffer.")

    def remove_scenario_id(self, scenario_id: str) -> bool:
        """Remove an unusable run-local scenario without altering frozen data."""

        target = str(scenario_id)
        before = len(self._records)
        self._records = [record for record in self._records if record.scenario_id != target]
        if len(self._records) != before:
            self._refresh_ranks()
            return True
        return False

    def sample_replay(
        self,
        *,
        rng: np.random.Generator,
        current_step: int,
        cfg: ScenarioAclReplaySamplingConfig,
        use_staleness: bool,
    ) -> ScenarioReplaySelection:
        if not self._records:
            raise ValueError("Cannot sample replay from an empty scenario buffer.")
        probabilities = compute_replay_probabilities(
            self._records,
            current_step=current_step,
            cfg=cfg,
            use_staleness=use_staleness,
        )
        index = int(rng.choice(len(self._records), p=probabilities))
        return ScenarioReplaySelection(
            index=index,
            record=self._records[index],
            probabilities=[float(value) for value in probabilities.tolist()],
        )

    def top_k(self, count: int) -> list[ScenarioRecord]:
        return assign_ranks(self._records)[: max(int(count), 0)]

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self._capacity,
            "size": len(self._records),
            "records": [record.to_dict() for record in self._records],
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "ScenarioBuffer":
        buffer = cls(capacity=int(payload.get("capacity", 1)))
        records_payload = payload.get("records", [])
        if not isinstance(records_payload, Iterable):
            raise TypeError("ScenarioBuffer state 'records' must be iterable.")
        for record_payload in records_payload:
            if not isinstance(record_payload, dict):
                raise TypeError("ScenarioBuffer state record entries must be mappings.")
            buffer._records.append(ScenarioRecord.from_dict(record_payload))
        buffer._refresh_ranks()
        return buffer
