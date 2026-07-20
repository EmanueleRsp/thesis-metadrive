"""Loading and validation helpers for immutable golden-suite manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GOLDEN_SUITE_SCHEMA = "scenarionet_golden_suite_content_validated_v1"


def load_golden_scenario_uids(path: str | Path) -> tuple[str, ...]:
    """Load the ordered scenario UID sequence from a validated manifest.

    The manifest is reference-only. This function reads it without copying or
    modifying any ScenarioNet source or runtime artifact.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Golden-suite manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Golden-suite manifest must contain a JSON object.")
    if payload.get("schema") != GOLDEN_SUITE_SCHEMA:
        raise ValueError(
            "Golden-suite manifest must be content-validated with schema "
            f"{GOLDEN_SUITE_SCHEMA!r}."
        )
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("Golden-suite manifest must contain a non-empty references list.")

    scenario_uids: list[str] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise ValueError(f"Golden-suite reference {index} must be an object.")
        scenario_uid = str(reference.get("scenario_uid", "")).strip()
        if not scenario_uid:
            raise ValueError(f"Golden-suite reference {index} has no scenario_uid.")
        if scenario_uid in scenario_uids:
            raise ValueError(f"Golden-suite manifest contains duplicate UID: {scenario_uid!r}")
        scenario_uids.append(scenario_uid)
    return tuple(scenario_uids)
