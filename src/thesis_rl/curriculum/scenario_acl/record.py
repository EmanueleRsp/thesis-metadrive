from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ScenarioRecord:
    scenario_id: str
    source: str
    parent_id: str | None
    scenario_description_path: str
    scenario_description_hash: str
    dataset_directory: str
    scenario_index: int
    env_config: dict[str, Any]
    reset_seed: int
    generator_arm: str | None
    mutation_type: str | None
    mutation_params: dict[str, Any] | None
    validation_status: str
    rule_criticality: float
    learning_potential: float
    usefulness: float
    usefulness_norm: float
    rank: int
    num_seen: int
    last_seen_step: int
    num_children: int
    metrics_summary: dict[str, Any]
    scenario_arm: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioRecord":
        data = dict(payload)
        data["scenario_index"] = int(data.get("scenario_index", 0))
        data["reset_seed"] = int(data.get("reset_seed", 0))
        data["rule_criticality"] = float(data.get("rule_criticality", 0.0))
        data["learning_potential"] = float(data.get("learning_potential", 0.0))
        data["usefulness"] = float(data.get("usefulness", 0.0))
        data["usefulness_norm"] = float(data.get("usefulness_norm", 0.0))
        data["rank"] = int(data.get("rank", 0))
        data["num_seen"] = int(data.get("num_seen", 0))
        data["last_seen_step"] = int(data.get("last_seen_step", 0))
        data["num_children"] = int(data.get("num_children", 0))
        data["env_config"] = dict(data.get("env_config", {}))
        data["metrics_summary"] = dict(data.get("metrics_summary", {}))
        mutation_params = data.get("mutation_params")
        if mutation_params is not None:
            data["mutation_params"] = dict(mutation_params)
        return cls(**data)
