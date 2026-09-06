"""Rulebook provenance must be recorded identically wherever it is written.

Regression cover for `open_items` C8: `conf/config.yaml` declared only
`rulebook.version`, and the two readers of that block substituted *different*
values for the undeclared keys -- `runtime/io/metadata.py` the literal ``"v1"``,
`contracts/reward_semantics.py` the version string. A single run therefore wrote
two contradictory families into `artifacts/run_metadata.yaml` and its own
checkpoint sidecar, and neither was the family that actually ran.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import yaml

from thesis_rl.contracts.reward_semantics import (
    RULEBOOK_PROVENANCE_FIELDS,
    UNDECLARED_RULEBOOK_PROVENANCE,
    build_reward_semantics_identity,
    rulebook_provenance,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPOSITORY_ROOT / "conf" / "config.yaml"

# `runtime/wiring/builders.py` routes this family string to the v2 adapter.
EXPECTED_IMPLEMENTATION_FAMILY = "v2"
EXPECTED_SPECIFICATION_ID = "RULEBOOK-V5.1"


def _shipped_rulebook_block() -> dict[str, object]:
    return dict(yaml.safe_load(SHIPPED_CONFIG.read_text())["rulebook"])


def test_shipped_configuration_declares_every_provenance_field() -> None:
    """The defect was an *omission*, so the fix is that nothing is omitted."""

    block = _shipped_rulebook_block()
    missing = [field for field in RULEBOOK_PROVENANCE_FIELDS if block.get(field) is None]
    assert not missing, f"conf/config.yaml leaves rulebook provenance undeclared: {missing}"


def test_shipped_configuration_names_the_rulebook_that_actually_runs() -> None:
    """Declaring a field is not enough if the declared value is the wrong one."""

    block = _shipped_rulebook_block()
    assert block["implementation_family"] == EXPECTED_IMPLEMENTATION_FAMILY
    assert block["specification_id"] == EXPECTED_SPECIFICATION_ID


@pytest.mark.parametrize(
    "rulebook_block",
    [
        pytest.param({}, id="nothing-declared"),
        pytest.param({"version": "4.7-final-implementation-complete"}, id="version-only"),
        pytest.param(
            {"implementation_family": "v2", "specification_id": "RULEBOOK-V5.1"},
            id="version-missing",
        ),
    ],
)
def test_both_writers_agree_on_an_underdeclared_block(rulebook_block: dict[str, str]) -> None:
    """The two readers must not diverge on a key neither of them was given.

    This is the assertion that fails on the pre-fix code: the metadata writer
    substituted ``"v1"`` where the identity builder substituted the version
    string, so the same input produced two different families.
    """

    config = {
        "reward": {"behavior": "scalar_reward"},
        "rulebook": dict(rulebook_block),
        "scalarization": {
            "specification_id": "SCAL-V1.4",
            "version": "1.4",
            "mode": "six_level_priority_weighted_rank",
            "vector_schema_id": "rulebook_v5_1_six_level_v1",
            "priority_base": 2.2,
            "numerical_tolerance": 1.0e-8,
            "native_environment_reward_weight": 0.0,
            "legacy": {"vector_schema_id": None, "rule_scales": None},
        },
    }

    identity = build_reward_semantics_identity(config)
    assert identity is not None

    from thesis_rl.runtime.io.metadata import _cfg_get

    cfg = OmegaConf.create(config)
    metadata_view = rulebook_provenance(
        {field: _cfg_get(cfg, f"rulebook.{field}") for field in RULEBOOK_PROVENANCE_FIELDS}
    )

    assert metadata_view == identity["rulebook"]
    for field in RULEBOOK_PROVENANCE_FIELDS:
        if field not in rulebook_block:
            assert metadata_view[field] == UNDECLARED_RULEBOOK_PROVENANCE


def test_undeclared_field_is_a_visible_sentinel_not_a_substantive_guess() -> None:
    """An absent key must read as absent, never as a plausible wrong answer."""

    resolved = rulebook_provenance({"version": "4.7-final-implementation-complete"})
    assert resolved["implementation_family"] == UNDECLARED_RULEBOOK_PROVENANCE
    assert resolved["specification_id"] == UNDECLARED_RULEBOOK_PROVENANCE
    assert resolved["version"] == "4.7-final-implementation-complete"
    assert resolved["implementation_family"] != "v1"


def test_declared_values_are_passed_through_unchanged() -> None:
    """The sentinel must not shadow a declaration."""

    declared = {
        "implementation_family": "v2",
        "specification_id": "RULEBOOK-V5.1",
        "version": "4.7-final-implementation-complete",
    }
    assert rulebook_provenance(declared) == declared
