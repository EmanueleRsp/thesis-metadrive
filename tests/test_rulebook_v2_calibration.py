from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.calibration import (
    BrakingTrial,
    calibrate_ego_braking,
    load_calibration_artifact,
    write_calibration_artifact,
)


def _trials(value: float = 3.27) -> list[BrakingTrial]:
    return [
        BrakingTrial(target, target, False, False, value)
        for target in (5.0, 10.0, 15.0, 20.0)
        for _ in range(10)
    ]


def test_calibration_uses_lower_quantile_floor_and_cap():
    trials = _trials()
    trials[0] = BrakingTrial(5.0, 5.0, False, False, 1.01)
    trials[1] = BrakingTrial(5.0, 5.0, False, False, 1.02)
    artifact = calibrate_ego_braking(trials=trials, config_hash="ego-hash")
    assert artifact.config_hash == "ego-hash"
    assert artifact.ego_min_brake_mps2 == pytest.approx(1.0)

    capped = calibrate_ego_braking(trials=_trials(6.0), config_hash="ego-hash")
    assert capped.ego_min_brake_mps2 == pytest.approx(4.0)


def test_calibration_rejects_unreachable_ego_and_insufficient_valid_trials():
    unreachable = [
        BrakingTrial(target, min(target, 19.0), False, False, 3.0)
        for target in (5.0, 10.0, 15.0, 20.0)
        for _ in range(10)
    ]
    with pytest.raises(ValueError, match="20 m/s"):
        calibrate_ego_braking(trials=unreachable, config_hash="ego-hash")

    invalid = _trials()
    invalid[0] = BrakingTrial(5.0, 5.5, False, False, 3.0)
    with pytest.raises(ValueError, match="ten valid"):
        calibrate_ego_braking(trials=invalid, config_hash="ego-hash")


def test_calibration_rejects_missing_hash_and_invalid_trial_values():
    with pytest.raises(ValueError, match="finite"):
        BrakingTrial(5.0, float("nan"), False, False, 3.0)
    with pytest.raises(ValueError, match="hash"):
        calibrate_ego_braking(trials=_trials(), config_hash="")


def test_calibration_artifact_roundtrip_validates_config_hash(tmp_path):
    artifact = calibrate_ego_braking(trials=_trials(), config_hash="ego-hash")
    path = write_calibration_artifact(artifact, tmp_path / "calibration.json")
    loaded = load_calibration_artifact(path, expected_config_hash="ego-hash")
    assert loaded == artifact
    with pytest.raises(ValueError, match="does not match"):
        load_calibration_artifact(path, expected_config_hash="other-hash")
