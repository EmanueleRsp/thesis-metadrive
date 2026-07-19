from __future__ import annotations

from pathlib import Path

import yaml


def test_rulebook_v2_config_targets_authoritative_v47() -> None:
    path = Path(__file__).parents[1] / "conf" / "rulebook" / "v2.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["version"] == "4.7-final-implementation-complete"
