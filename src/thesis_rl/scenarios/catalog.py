from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


_RECORD_FIELDS = {field.name for field in fields(ScenarioRecord)}
_FEATURE_FIELDS = {field.name for field in fields(ScenarioFeatures)}
_SHARED_FIELDS = {"scenario_id", "source", "length"}


@dataclass(frozen=True, slots=True)
class ScenarioCatalogEntry:
    record: ScenarioRecord
    features: ScenarioFeatures

    def __post_init__(self) -> None:
        for name in _SHARED_FIELDS:
            if getattr(self.record, name) != getattr(self.features, name):
                raise ValueError(f"record/features mismatch for {name}")

    def to_flat_dict(self) -> dict[str, Any]:
        payload = self.record.to_dict()
        feature_payload = self.features.to_dict()
        payload.update(
            {key: value for key, value in feature_payload.items() if key not in _SHARED_FIELDS}
        )
        return payload

    @classmethod
    def from_flat_dict(cls, payload: dict[str, Any]) -> "ScenarioCatalogEntry":
        record_payload = {key: payload.get(key) for key in _RECORD_FIELDS}
        feature_payload = {key: payload.get(key) for key in _FEATURE_FIELDS}
        return cls(
            record=ScenarioRecord.from_dict(record_payload),
            features=ScenarioFeatures.from_dict(feature_payload),
        )


class ScenarioCatalog:
    def __init__(self, entries: Sequence[ScenarioCatalogEntry]) -> None:
        self._entries = tuple(entries)
        uid_index: dict[str, ScenarioCatalogEntry] = {}
        runtime_index: dict[tuple[str, int], ScenarioCatalogEntry] = {}
        for entry in self._entries:
            record = entry.record
            if record.scenario_uid in uid_index:
                raise ValueError(f"duplicate scenario_uid: {record.scenario_uid}")
            uid_index[record.scenario_uid] = entry
            if record.runtime_index is not None:
                key = (record.split, record.runtime_index)
                if key in runtime_index:
                    raise ValueError(
                        f"duplicate runtime_index {record.runtime_index} in split {record.split}"
                    )
                runtime_index[key] = entry
        self._uid_index = uid_index
        self._runtime_index = runtime_index

    @property
    def entries(self) -> tuple[ScenarioCatalogEntry, ...]:
        return self._entries

    @property
    def records(self) -> tuple[ScenarioRecord, ...]:
        return tuple(entry.record for entry in self._entries)

    def get_by_uid(self, scenario_uid: str) -> ScenarioCatalogEntry:
        try:
            return self._uid_index[scenario_uid]
        except KeyError as exc:
            raise KeyError(f"unknown scenario_uid: {scenario_uid}") from exc

    def get_by_runtime_index(self, *, split: str, runtime_index: int) -> ScenarioCatalogEntry:
        try:
            return self._runtime_index[(split, int(runtime_index))]
        except KeyError as exc:
            raise KeyError(
                f"unknown runtime_index {runtime_index} for split {split!r}"
            ) from exc

    def valid_records(self, *, split: str | None = None) -> tuple[ScenarioRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.validation_status in {"valid", "warning"}
            and (split is None or record.split == split)
        )


def write_scenario_catalog(
    entries: Iterable[ScenarioCatalogEntry],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    catalog = ScenarioCatalog(tuple(entries))
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite scenario catalog: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [entry.to_flat_dict() for entry in catalog.entries]
    if not rows:
        raise ValueError("scenario catalog cannot be empty")
    table = pa.Table.from_pylist(rows)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(target)
    return target


def read_scenario_catalog(path: str | Path) -> ScenarioCatalog:
    table = pq.read_table(Path(path))
    entries = [ScenarioCatalogEntry.from_flat_dict(row) for row in table.to_pylist()]
    return ScenarioCatalog(entries)
