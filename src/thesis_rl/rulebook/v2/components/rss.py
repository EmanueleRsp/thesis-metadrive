"""R2 longitudinal RSS evaluator with validated calibration artifact."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.types import CacheDelta, ComponentStatus, MemoryDelta, RuleComponentResult


@dataclass(frozen=True, slots=True)
class RSSCalibrationArtifact:
    config_hash: str
    ego_min_brake_mps2: float

    def __post_init__(self) -> None:
        if not self.config_hash or not isfinite(self.ego_min_brake_mps2) or self.ego_min_brake_mps2 <= 0.0:
            raise ValueError("RSS calibration artifact must contain a positive finite brake value and hash")


@dataclass(frozen=True, slots=True)
class RSSCandidate:
    actor_id: str
    gap_m: float
    ego_speed_mps: float
    front_speed_mps: float


RESPONSE_TIME_S = 1.0
MAX_RESPONSE_ACCEL_MPS2 = 3.5
FRONT_MAX_BRAKE_MPS2 = 8.0


def _fail(scenario_id: str, step_index: int, cause: str) -> None:
    raise RulebookEvaluationError(EvaluationFailure(scenario_id, step_index, "rss", cause))


def safe_distance_m(*, ego_speed_mps: float, front_speed_mps: float, ego_brake_mps2: float) -> float:
    if not all(isfinite(value) for value in (ego_speed_mps, front_speed_mps, ego_brake_mps2)):
        raise ValueError("RSS speeds and braking must be finite")
    if ego_brake_mps2 <= 0.0:
        raise ValueError("RSS ego braking must be positive")
    ego_speed = max(0.0, ego_speed_mps)
    front_speed = max(0.0, front_speed_mps)
    response_distance = ego_speed * RESPONSE_TIME_S
    response_acceleration = 0.5 * MAX_RESPONSE_ACCEL_MPS2 * RESPONSE_TIME_S**2
    ego_braking = (ego_speed + RESPONSE_TIME_S * MAX_RESPONSE_ACCEL_MPS2) ** 2 / (2.0 * ego_brake_mps2)
    front_braking = front_speed**2 / (2.0 * FRONT_MAX_BRAKE_MPS2)
    return max(0.0, response_distance + response_acceleration + ego_braking - front_braking)


def evaluate_rss(
    *,
    scenario_id: str,
    step_index: int,
    candidates: tuple[RSSCandidate, ...],
    calibration: RSSCalibrationArtifact | None,
    expected_config_hash: str,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate front vehicles; ambiguous/no-front context is NOT_APPLICABLE."""

    if candidates and calibration is None:
        _fail(scenario_id, step_index, "RSS calibration artifact is missing")
    if candidates and calibration.config_hash != expected_config_hash:
        _fail(scenario_id, step_index, "RSS calibration artifact hash does not match ego config")
    if not candidates:
        return (
            RuleComponentResult(
                "rss", 0.0, {"actors": ()}, False, True,
                ComponentStatus.NOT_APPLICABLE, {"candidate_count": 0}
            ),
            MemoryDelta(), CacheDelta(),
        )
    values: list[tuple[str, float, float, float]] = []
    for candidate in candidates:
        if not isfinite(candidate.gap_m) or candidate.gap_m < 0.0:
            _fail(scenario_id, step_index, f"RSS gap for actor {candidate.actor_id!r} is invalid")
        safe = safe_distance_m(
            ego_speed_mps=candidate.ego_speed_mps,
            front_speed_mps=candidate.front_speed_mps,
            ego_brake_mps2=calibration.ego_min_brake_mps2,
        )
        deficit = max(0.0, safe - candidate.gap_m)
        cost = 0.0 if safe == 0.0 else max(0.0, 1.0 - candidate.gap_m / safe)
        values.append((candidate.actor_id, safe, deficit, cost))
    worst_actor, safe, deficit, cost = max(values, key=lambda value: (value[3], value[0]))
    result = RuleComponentResult(
        "rss", cost,
        {
            "worst_actor_id": worst_actor,
            "worst_safe_distance_m": safe,
            "worst_deficit_m": deficit,
            "actors": tuple(
                {"actor_id": actor_id, "safe_distance_m": actor_safe, "deficit_m": actor_deficit, "cost": actor_cost}
                for actor_id, actor_safe, actor_deficit, actor_cost in values
            ),
        },
        True, True,
        ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        {"candidate_count": len(values)},
    )
    return result, MemoryDelta(), CacheDelta()
