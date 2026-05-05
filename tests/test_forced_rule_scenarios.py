from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_forced_rule_scenarios_activate_expected_rules(tmp_path: Path) -> None:
    out_path = tmp_path / "forced_rule_scenarios.json"

    cmd = [
        sys.executable,
        "src/thesis_rl/tools/debug/force_rule_scenarios.py",
        "--out",
        str(out_path),
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    subprocess.run(cmd, check=True, env=env)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    scenarios = {item["label"]: item for item in payload.get("scenarios", [])}

    assert "baseline_after_reset" in scenarios
    assert "forced_collision" in scenarios
    assert "forced_wrong_way" in scenarios
    assert "forced_out_of_drivable" in scenarios

    baseline = scenarios["baseline_after_reset"]["rule_components"]
    collision = scenarios["forced_collision"]["rule_components"]
    wrong_way = scenarios["forced_wrong_way"]["rule_components"]
    out_of_drivable = scenarios["forced_out_of_drivable"]["rule_components"]

    assert float(baseline["vehicle_collision_energy"]) == 0.0
    assert float(baseline["wrong_way"]) == 0.0
    assert float(baseline["drivable_area"]) == 0.0

    assert float(collision["vehicle_collision_energy"]) < 0.0
    assert float(wrong_way["wrong_way"]) < 0.0
    assert float(out_of_drivable["drivable_area"]) < 0.0
