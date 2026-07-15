"""Build the normative Rulebook v2 ego-braking calibration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.calibration import BrakingTrial, calibrate_ego_braking, write_calibration_artifact


def _read_trials(path: Path) -> tuple[BrakingTrial, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trial file is not readable JSON") from error
    raw_trials: Any = payload.get("trials") if isinstance(payload, dict) else payload
    if not isinstance(raw_trials, list):
        raise ValueError("trial JSON must be a list or an object containing 'trials'")
    trials: list[BrakingTrial] = []
    for index, raw in enumerate(raw_trials):
        if not isinstance(raw, dict):
            raise ValueError(f"trial {index} must be an object")
        try:
            trials.append(
                BrakingTrial(
                    target_speed_mps=float(raw["target_speed_mps"]),
                    reached_speed_mps=float(raw["reached_speed_mps"]),
                    collided=bool(raw["collided"]),
                    left_lane=bool(raw["left_lane"]),
                    mean_deceleration_mps2=float(raw["mean_deceleration_mps2"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"trial {index} is invalid") from error
    return tuple(trials)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="JSON file with braking trials")
    parser.add_argument("--config-hash", required=True, help="Hash of the frozen ego configuration")
    parser.add_argument("--out", type=Path, required=True, help="Output calibration artifact JSON")
    args = parser.parse_args()

    trials = _read_trials(args.trials.expanduser().resolve())
    artifact = calibrate_ego_braking(trials=trials, config_hash=args.config_hash)
    output = write_calibration_artifact(artifact, args.out.expanduser().resolve())
    print(
        json.dumps(
            {
                "artifact": str(output),
                "config_hash": artifact.config_hash,
                "ego_min_brake_mps2": artifact.ego_min_brake_mps2,
                "trial_count": len(trials),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
