"""Normative one-shot calibration protocol for the ego RSS braking value."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import floor, isfinite
from pathlib import Path
from typing import Sequence

from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact


CALIBRATION_TARGET_SPEEDS_MPS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
MIN_VALID_TRIALS_PER_SPEED = 10
TARGET_SPEED_TOLERANCE_MPS = 0.2
# Physical bound: dry-asphalt tyre-road deceleration limit. It is also the
# value already assumed for surrounding vehicles (see
# components/rss.py:FRONT_MAX_BRAKE_MPS2), so the RSS model no longer assumes
# the ego brakes worse than identical traffic. Measured ego braking (12-trial
# sample, 2026-08-01) is 10.73-16.92 m/s2, so this bound remains conservative
# relative to the measurement while removing the previous factor-of-2.7 cap.
MAX_REFERENCE_BRAKE_MPS2 = 8.0
CALIBRATION_ARTIFACT_SCHEMA = "rulebook-v2-braking-calibration-v1"


@dataclass(frozen=True, slots=True)
class BrakingTrial:
    """Measured result of one straight-road maximum-braking trial."""

    target_speed_mps: float
    reached_speed_mps: float
    collided: bool
    left_lane: bool
    mean_deceleration_mps2: float

    def __post_init__(self) -> None:
        values = (
            self.target_speed_mps,
            self.reached_speed_mps,
            self.mean_deceleration_mps2,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Braking trial speeds and deceleration must be finite")
        if self.target_speed_mps not in CALIBRATION_TARGET_SPEEDS_MPS:
            raise ValueError(f"Unsupported calibration target speed: {self.target_speed_mps}")
        if self.reached_speed_mps < 0.0 or self.mean_deceleration_mps2 <= 0.0:
            raise ValueError("Braking trial reached speed and deceleration must be positive")


def _lower_quantile(values: Sequence[float], probability: float) -> float:
    """Order-statistic quantile with NumPy's ``method='lower'`` semantics."""

    if not values:
        raise ValueError("Cannot calculate a quantile from no values")
    ordered = sorted(values)
    index = floor(probability * (len(ordered) - 1))
    return float(ordered[index])


def calibrate_ego_braking(
    *,
    trials: Sequence[BrakingTrial],
    config_hash: str,
) -> RSSCalibrationArtifact:
    """Produce the shared RSS/signal/crosswalk/vehicle-yield brake artifact.

    Trial validity follows the frozen protocol: the target must be reached
    within 0.2 m/s, with no collision or lane exit.  Invalid trials are
    discarded, but malformed/non-finite records fail fast before aggregation.
    """

    if not config_hash:
        raise ValueError("Calibration requires a non-empty ego configuration hash")
    trial_list = tuple(trials)
    if not trial_list:
        raise ValueError("Calibration requires braking trials")
    if max(trial.reached_speed_mps for trial in trial_list) < 20.0:
        raise ValueError("Ego configuration did not reach the required 20 m/s")

    valid_by_target: dict[float, list[float]] = {
        target: [] for target in CALIBRATION_TARGET_SPEEDS_MPS
    }
    for trial in trial_list:
        valid = (
            abs(trial.reached_speed_mps - trial.target_speed_mps) <= TARGET_SPEED_TOLERANCE_MPS
            and not trial.collided
            and not trial.left_lane
        )
        if valid:
            valid_by_target[trial.target_speed_mps].append(trial.mean_deceleration_mps2)
    missing = tuple(
        target
        for target, values in valid_by_target.items()
        if len(values) < MIN_VALID_TRIALS_PER_SPEED
    )
    if missing:
        raise ValueError(f"Calibration requires ten valid trials per target; missing: {missing}")

    measured = tuple(value for values in valid_by_target.values() for value in values)
    b_meas = _lower_quantile(measured, 0.05)
    calibrated = min(MAX_REFERENCE_BRAKE_MPS2, floor(10.0 * b_meas) / 10.0)
    if not isfinite(calibrated) or calibrated <= 0.0:
        raise ValueError("Calibration produced a non-positive or non-finite ego brake value")
    return RSSCalibrationArtifact(config_hash=config_hash, ego_min_brake_mps2=calibrated)


def write_calibration_artifact(
    artifact: RSSCalibrationArtifact,
    path: str | Path,
) -> Path:
    """Persist a minimal, auditable calibration artifact as canonical JSON."""

    if not isinstance(artifact, RSSCalibrationArtifact):
        raise TypeError("artifact must be an RSSCalibrationArtifact")
    if artifact.ego_min_brake_mps2 > MAX_REFERENCE_BRAKE_MPS2:
        raise ValueError("Calibration artifact brake value exceeds the normative cap")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CALIBRATION_ARTIFACT_SCHEMA,
        "config_hash": artifact.config_hash,
        "ego_min_brake_mps2": artifact.ego_min_brake_mps2,
        "target_speeds_mps": CALIBRATION_TARGET_SPEEDS_MPS,
        "quantile": "lower_0.05",
        "rounding": "floor_0.1",
        "cap_mps2": MAX_REFERENCE_BRAKE_MPS2,
    }
    output_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_calibration_artifact(
    path: str | Path,
    *,
    expected_config_hash: str,
) -> RSSCalibrationArtifact:
    """Load and validate the persisted artifact before a final run."""

    if not expected_config_hash:
        raise ValueError("Expected calibration config hash must be non-empty")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Calibration artifact is not readable JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != CALIBRATION_ARTIFACT_SCHEMA:
        raise ValueError("Calibration artifact schema is invalid")
    if payload.get("config_hash") != expected_config_hash:
        raise ValueError("Calibration artifact hash does not match ego config")
    try:
        target_speeds = tuple(float(value) for value in payload["target_speeds_mps"])
        cap_mps2 = float(payload["cap_mps2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration artifact protocol metadata is invalid") from error
    if target_speeds != CALIBRATION_TARGET_SPEEDS_MPS or cap_mps2 != MAX_REFERENCE_BRAKE_MPS2:
        raise ValueError("Calibration artifact protocol metadata is invalid")
    try:
        artifact = RSSCalibrationArtifact(
            config_hash=str(payload["config_hash"]),
            ego_min_brake_mps2=float(payload["ego_min_brake_mps2"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration artifact values are invalid") from error
    if payload.get("quantile") != "lower_0.05" or payload.get("rounding") != "floor_0.1":
        raise ValueError("Calibration artifact protocol metadata is invalid")
    if artifact.ego_min_brake_mps2 > MAX_REFERENCE_BRAKE_MPS2:
        raise ValueError("Calibration artifact brake value exceeds the normative cap")
    return artifact
