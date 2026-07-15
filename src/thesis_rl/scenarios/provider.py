from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Collection, Sequence

import numpy as np

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.records import ScenarioRecord


class ScenarioProvider(ABC):
    @abstractmethod
    def sample(
        self,
        *,
        split: str,
        worker_id: int,
        source: str | None = None,
        arm: str | None = None,
        excluded_scenario_uids: Collection[str] = (),
    ) -> ScenarioRecord:
        raise NotImplementedError


def _valid_records(records: Sequence[ScenarioRecord], eligible_scenario_uids: Collection[str] | None = None) -> tuple[ScenarioRecord, ...]:
    eligible = None if eligible_scenario_uids is None else {str(value) for value in eligible_scenario_uids}
    return tuple(
        record for record in records
        if record.validation_status in {"valid", "warning"}
        and (eligible is None or record.scenario_uid in eligible)
    )


class UniformScenarioProvider(ScenarioProvider):
    def __init__(
        self,
        records: Sequence[ScenarioRecord],
        *,
        global_seed: int,
        source_probabilities: dict[str, float] | None = None,
        strict: bool = True,
        allow_fallback: bool = False,
        default_arm: str | None = None,
        eligible_scenario_uids: Collection[str] | None = None,
    ) -> None:
        if not strict or allow_fallback:
            raise ValueError("ScenarioNet v1 provider requires strict=true and allow_fallback=false")
        self._records = _valid_records(records, eligible_scenario_uids)
        if not self._records:
            raise ValueError("provider requires at least one valid scenario")
        probabilities = dict(source_probabilities or {"waymo": 0.5, "pg": 0.5})
        if set(probabilities) != {"waymo", "pg"}:
            raise ValueError("source probabilities must define exactly waymo and pg")
        if any(value < 0 for value in probabilities.values()) or not np.isclose(
            sum(probabilities.values()), 1.0
        ):
            raise ValueError("source probabilities must be non-negative and sum to one")
        self._source_names = ("waymo", "pg")
        self._source_probabilities = np.asarray(
            [probabilities[name] for name in self._source_names], dtype=np.float64
        )
        self._global_seed = int(global_seed)
        self._default_arm = _validate_arm(default_arm)
        self._rng_by_worker: dict[int, np.random.Generator] = {}
        self.reset_counts: dict[str, int] = defaultdict(int)

    def _rng(self, worker_id: int) -> np.random.Generator:
        worker = int(worker_id)
        if worker < 0:
            raise ValueError("worker_id must be non-negative")
        if worker not in self._rng_by_worker:
            seed_sequence = np.random.SeedSequence([self._global_seed, worker])
            self._rng_by_worker[worker] = np.random.default_rng(seed_sequence)
        return self._rng_by_worker[worker]

    def sample(
        self,
        *,
        split: str,
        worker_id: int,
        source: str | None = None,
        arm: str | None = None,
        excluded_scenario_uids: Collection[str] = (),
    ) -> ScenarioRecord:
        rng = self._rng(worker_id)
        requested_arm = _validate_arm(arm) if arm is not None else self._default_arm
        excluded = {str(value) for value in excluded_scenario_uids}
        candidates = [
            record
            for record in self._records
            if record.split == split
            and record.scenario_uid not in excluded
            and (requested_arm is None or record.primary_arm == requested_arm)
        ]
        if source is not None:
            if source not in self._source_names:
                raise ValueError(f"unsupported source: {source!r}")
            candidates = [record for record in candidates if record.source == source]
        elif candidates:
            available_sources = {record.source for record in candidates}
            source_indices = [
                index for index, name in enumerate(self._source_names)
                if name in available_sources
            ]
            probabilities = self._source_probabilities[source_indices]
            probabilities = probabilities / probabilities.sum()
            selected_index = int(rng.choice(source_indices, p=probabilities))
            selected_source = self._source_names[selected_index]
            candidates = [record for record in candidates if record.source == selected_source]
        if not candidates:
            filters = f"split={split!r}, source={source!r}, arm={requested_arm!r}"
            raise LookupError(f"no valid scenarios for {filters}; fallback is disabled")
        selected = candidates[int(rng.integers(0, len(candidates)))]
        self.reset_counts[selected.source] += 1
        return selected

    def has_candidate(
        self,
        *,
        split: str,
        source: str | None = None,
        arm: str | None = None,
        excluded_scenario_uids: Collection[str] = (),
    ) -> bool:
        requested_arm = _validate_arm(arm) if arm is not None else self._default_arm
        excluded = {str(value) for value in excluded_scenario_uids}
        return any(
            record.split == split
            and record.scenario_uid not in excluded
            and (source is None or record.source == source)
            and (requested_arm is None or record.primary_arm == requested_arm)
            for record in self._records
        )


class FixedSequenceScenarioProvider(ScenarioProvider):
    def __init__(
        self,
        records: Sequence[ScenarioRecord],
        *,
        repeat: bool = False,
        default_arm: str | None = None,
        eligible_scenario_uids: Collection[str] | None = None,
    ) -> None:
        self._records = _valid_records(records, eligible_scenario_uids)
        if eligible_scenario_uids is None and len(self._records) != len(records):
            raise ValueError("fixed sequence contains invalid scenarios")
        if not self._records:
            raise ValueError("fixed sequence cannot be empty")
        if len({record.scenario_uid for record in self._records}) != len(self._records):
            raise ValueError("fixed sequence contains duplicate scenario_uid values")
        self._repeat = bool(repeat)
        self._default_arm = _validate_arm(default_arm)
        self._position = 0

    def sample(
        self,
        *,
        split: str,
        worker_id: int,
        source: str | None = None,
        arm: str | None = None,
        excluded_scenario_uids: Collection[str] = (),
    ) -> ScenarioRecord:
        if worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if self._position >= len(self._records):
            if not self._repeat:
                raise LookupError("fixed scenario sequence is exhausted")
            self._position = 0
        record = self._records[self._position]
        if record.scenario_uid in {str(value) for value in excluded_scenario_uids}:
            raise LookupError("fixed sequence record is excluded from fresh sampling")
        if record.split != split:
            raise LookupError(
                f"fixed sequence record {record.scenario_uid} belongs to {record.split}, not {split}"
            )
        if source is not None and record.source != source:
            raise LookupError(f"fixed sequence record does not match requested source {source!r}")
        requested_arm = _validate_arm(arm) if arm is not None else self._default_arm
        if requested_arm is not None and record.primary_arm != requested_arm:
            raise LookupError(
                f"fixed sequence record does not match requested arm {requested_arm!r}"
            )
        self._position += 1
        return record


def _validate_arm(arm: str | None) -> str | None:
    if arm is None:
        return None
    value = str(arm).strip()
    if value not in ARMS:
        raise ValueError(f"unsupported scenario arm: {value!r}; expected one of {ARMS}")
    return value
