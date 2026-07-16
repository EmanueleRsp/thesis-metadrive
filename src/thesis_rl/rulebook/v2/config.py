"""Immutable Rulebook v2 configuration and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from typing import Mapping

from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.types import MACRO_RULE_ORDER, MacroRule


RULEBOOK_V2_VERSION = "4.6-final-implementation-complete"


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    precision_grid_m: float = 1.0e-3
    offroad_area_epsilon_m2: float = 1.0e-4
    polyline_buffer_epsilon_m: float = 1.0e-2
    signed_distance_epsilon_m: float = 5.0e-2
    lane_angle_equivalence_epsilon_rad: float = 1.0e-6
    lane_lateral_equivalence_epsilon_m: float = 1.0e-3
    vertical_compatibility_tolerance_m: float = 3.0
    interval_time_epsilon_s: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    horizon_s: float = 3.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    evaluator_side_effects: str = "forbidden"
    memory_and_cache_commit: str = "atomic_after_success"
    not_evaluable_policy: str = "fail_fast"
    silent_fallbacks: bool = False
    future_ground_truth_tracks: bool = False


@dataclass(frozen=True, slots=True)
class RulebookV2Config:
    version: str = RULEBOOK_V2_VERSION
    order: tuple[MacroRule, ...] = MACRO_RULE_ORDER
    execution: ExecutionConfig = ExecutionConfig()
    geometry: GeometryConfig = GeometryConfig()
    prediction: PredictionConfig = PredictionConfig()

    def validate(self) -> None:
        if self.version != RULEBOOK_V2_VERSION:
            raise ValueError(f"Unsupported Rulebook v2 version: {self.version!r}")
        if self.order != MACRO_RULE_ORDER:
            raise ValueError("Rulebook v2 macro-rule order is fixed by the specification.")
        if self.execution.evaluator_side_effects != "forbidden":
            raise ValueError("Rulebook v2 evaluators must be pure.")
        if self.execution.memory_and_cache_commit != "atomic_after_success":
            raise ValueError("Rulebook v2 memory/cache commit must be atomic after success.")
        if self.execution.not_evaluable_policy != "fail_fast" or self.execution.silent_fallbacks:
            raise ValueError("Rulebook v2 requires fail-fast NOT_EVALUABLE without fallbacks.")
        if self.execution.future_ground_truth_tracks:
            raise ValueError("Rulebook v2 cannot read future ground-truth tracks online.")
        for geometry_field in fields(self.geometry):
            field_name = geometry_field.name
            value = getattr(self.geometry, field_name)
            if value <= 0.0:
                raise ValueError(f"geometry.{field_name} must be positive, got {value!r}")
        if self.prediction.horizon_s != 3.0:
            raise ValueError("prediction.horizon_s is frozen at 3.0 s in Rulebook v2.")


def load_rulebook_v2_config(data: Mapping[str, object]) -> RulebookV2Config:
    """Load the v2-owned subset of YAML after rejecting unknown semantic knobs."""

    unsupported = set(data) - {"version"}
    if unsupported:
        raise ValueError(f"Unsupported Rulebook v2 config keys: {sorted(unsupported)!r}")
    version = str(data.get("version", RULEBOOK_V2_VERSION))
    config = RulebookV2Config(version=version)
    config.validate()
    return config


def geometry_config_hash(config: RulebookV2Config | None = None) -> str:
    """Return the canonical provenance hash for the frozen geometry contract."""

    resolved = config or RulebookV2Config()
    resolved.validate()
    payload = {
        "rulebook_version": resolved.version,
        "geometry": asdict(resolved.geometry),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fail_not_evaluable(*, scenario_id: str, step_index: int, component: str, cause: str) -> None:
    """Small common entry point used by future pure evaluators."""

    raise RulebookEvaluationError(EvaluationFailure(scenario_id, step_index, component, cause))
