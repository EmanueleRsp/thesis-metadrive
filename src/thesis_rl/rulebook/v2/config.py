"""Immutable Rulebook v2 configuration and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from math import isfinite
from typing import Mapping

from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.types import MACRO_RULE_ORDER, MacroRule


RULEBOOK_V2_VERSION = "4.7-final-implementation-complete"


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
class TTCConfig:
    motion_model: str = "constant_velocity_constant_heading_no_rotation"
    occupancy_solver: str = "continuous_sat_after_deterministic_convex_decomposition"


@dataclass(frozen=True, slots=True)
class ConflictZoneOccupancyConfig:
    vehicle_motion_model: str = "causal_ctrv_ols"
    non_vehicle_motion_model: str = "constant_velocity_constant_heading_no_rotation"
    history_window_s: float = 0.5
    minimum_history_samples: int = 3
    stationary_speed_epsilon_mps: float = 0.1
    yaw_rate_straight_epsilon_rad_s: float = 1.0e-3
    rotating_solver: str = "deterministic_uniform_sweep_with_bisection"
    rotating_occupancy_max_step_s: float = 0.02
    interval_selection: str = "containing_zero_else_earliest"
    open_end_representation: str = "symbolic"


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    horizon_s: float = 3.0
    ttc: TTCConfig = TTCConfig()
    conflict_zone_occupancy: ConflictZoneOccupancyConfig = ConflictZoneOccupancyConfig()


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
        if self.prediction.ttc.motion_model != "constant_velocity_constant_heading_no_rotation":
            raise ValueError("TTC motion model is frozen to v4.6 constant velocity.")
        if (
            self.prediction.ttc.occupancy_solver
            != "continuous_sat_after_deterministic_convex_decomposition"
        ):
            raise ValueError("TTC occupancy solver is frozen to exact v4.6 continuous SAT.")
        occupancy = self.prediction.conflict_zone_occupancy
        if occupancy.vehicle_motion_model != "causal_ctrv_ols":
            raise ValueError("Vehicle conflict-zone motion model is frozen to causal_ctrv_ols.")
        if occupancy.non_vehicle_motion_model != "constant_velocity_constant_heading_no_rotation":
            raise ValueError("Non-vehicle conflict-zone motion model must remain v4.6 CV.")
        if occupancy.rotating_solver != "deterministic_uniform_sweep_with_bisection":
            raise ValueError(
                "CTRV rotating solver is frozen to deterministic sweep with bisection."
            )
        if occupancy.interval_selection != "containing_zero_else_earliest":
            raise ValueError("CTRV interval selection is frozen by the specification.")
        if occupancy.open_end_representation != "symbolic":
            raise ValueError("CTRV OPEN_END representation is symbolic and frozen.")
        if occupancy.history_window_s != 0.5:
            raise ValueError("CTRV history_window_s is frozen at 0.5 s.")
        if occupancy.minimum_history_samples != 3:
            raise ValueError("CTRV minimum_history_samples is frozen at 3.")
        if (
            not isfinite(occupancy.history_window_s)
            or occupancy.history_window_s <= 0.0
            or occupancy.minimum_history_samples < 2
        ):
            raise ValueError("CTRV history configuration is out of range.")
        if occupancy.stationary_speed_epsilon_mps != 0.1:
            raise ValueError("stationary_speed_epsilon_mps is frozen at 0.1 m/s.")
        if (
            not isfinite(occupancy.stationary_speed_epsilon_mps)
            or occupancy.stationary_speed_epsilon_mps < 0.0
        ):
            raise ValueError("stationary_speed_epsilon_mps must be non-negative.")
        if occupancy.yaw_rate_straight_epsilon_rad_s != 1.0e-3:
            raise ValueError("yaw_rate_straight_epsilon_rad_s is frozen at 1e-3 rad/s.")
        if (
            not isfinite(occupancy.yaw_rate_straight_epsilon_rad_s)
            or occupancy.yaw_rate_straight_epsilon_rad_s <= 0.0
        ):
            raise ValueError("yaw_rate_straight_epsilon_rad_s must be positive.")
        if occupancy.rotating_occupancy_max_step_s != 0.02:
            raise ValueError("rotating_occupancy_max_step_s is frozen at 0.02 s.")
        if (
            not isfinite(occupancy.rotating_occupancy_max_step_s)
            or not 0.0 < occupancy.rotating_occupancy_max_step_s <= self.prediction.horizon_s
        ):
            raise ValueError("rotating_occupancy_max_step_s must be within the horizon.")


def load_rulebook_v2_config(data: Mapping[str, object]) -> RulebookV2Config:
    """Load the v2-owned subset of YAML after rejecting unknown semantic knobs."""

    unsupported = set(data) - {"version", "prediction"}
    if unsupported:
        raise ValueError(f"Unsupported Rulebook v2 config keys: {sorted(unsupported)!r}")
    version = str(data.get("version", RULEBOOK_V2_VERSION))
    prediction_data = data.get("prediction", {})
    if not isinstance(prediction_data, Mapping):
        raise ValueError("prediction must be a mapping")
    unsupported_prediction = set(prediction_data) - {"horizon_s", "ttc", "conflict_zone_occupancy"}
    if unsupported_prediction:
        raise ValueError(f"Unsupported prediction config keys: {sorted(unsupported_prediction)!r}")
    ttc_data = prediction_data.get("ttc", {})
    occupancy_data = prediction_data.get("conflict_zone_occupancy", {})
    if not isinstance(ttc_data, Mapping) or not isinstance(occupancy_data, Mapping):
        raise ValueError("prediction.ttc and prediction.conflict_zone_occupancy must be mappings")
    if set(ttc_data) - {"motion_model", "occupancy_solver"}:
        raise ValueError("Unsupported prediction.ttc config keys")
    if set(occupancy_data) - {
        "vehicle_motion_model",
        "non_vehicle_motion_model",
        "history_window_s",
        "minimum_history_samples",
        "stationary_speed_epsilon_mps",
        "yaw_rate_straight_epsilon_rad_s",
        "rotating_solver",
        "interval_selection",
        "open_end_representation",
        "rotating_occupancy_max_step_s",
    }:
        raise ValueError("Unsupported prediction.conflict_zone_occupancy config keys")
    ttc = TTCConfig(**{key: value for key, value in ttc_data.items()})
    occupancy = ConflictZoneOccupancyConfig(**{key: value for key, value in occupancy_data.items()})
    prediction = PredictionConfig(
        horizon_s=prediction_data.get("horizon_s", 3.0),
        ttc=ttc,
        conflict_zone_occupancy=occupancy,
    )
    config = RulebookV2Config(version=version, prediction=prediction)
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
