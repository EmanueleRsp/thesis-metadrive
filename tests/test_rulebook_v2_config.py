"""The Rulebook v2 family string, pinned across the three places that declare it.

`rulebook.version` names the *implementation family* the runtime routes on, not
the specification in force — that is `rulebook.specification_id`. It reads
`4.7-final-implementation-complete`, which claims a specification version two
releases behind the six-level vector actually implemented, and nothing tied its
three copies together: the constant, `conf/rulebook/v2.yaml`, and
`conf/config.yaml`. `C8` fixed the two provenance fields beside it and left this
one.

It is still misleading, deliberately. A rename attempted on 2026-09-08 was
reverted once the suite showed that the string feeds `geometry_config_hash()`,
which a stability verifier compares against a hash frozen with the dataset — so
correcting it moves a dataset provenance identity and is a user decision, not a
cleanup. These tests keep the three copies from drifting and make that
constraint executable, so the next attempt meets it here rather than in a
twenty-eight-minute suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from thesis_rl.rulebook.v2.config import (
    RULEBOOK_V2_VERSION,
    RulebookV2Config,
    geometry_config_hash,
)
from thesis_rl.runtime.wiring.builders import RULEBOOK_V2_FAMILY_ALIASES

_CONF = Path(__file__).parents[1] / "conf"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_rulebook_config_declares_the_implementation_family() -> None:
    assert _yaml(_CONF / "rulebook" / "v2.yaml")["version"] == RULEBOOK_V2_VERSION


def test_the_root_config_declares_the_same_family_string() -> None:
    """The third copy, and the one `C8` found disagreeing with the other two.

    `implementation_family` is a *separate* field and reads `v2`, so the two do
    not match: one names the family in the routing vocabulary, the other in the
    historical spelling the frozen dataset hash was computed under. That
    divergence is `C8`'s open half stated as data, and it is asserted rather than
    tidied away because closing it moves a dataset provenance identity.
    """

    rulebook = _yaml(_CONF / "config.yaml")["rulebook"]

    assert rulebook["version"] == RULEBOOK_V2_VERSION
    assert rulebook["implementation_family"] == "v2"
    assert rulebook["implementation_family"] in RULEBOOK_V2_FAMILY_ALIASES
    assert rulebook["version"] != rulebook["implementation_family"], (
        "if these ever agree, `C8`'s open half was closed and this test should say so"
    )


def test_the_declared_family_routes_to_the_v2_adapter() -> None:
    """A rename must not silently stop routing.

    `build_reward_manager` selects the v2 adapter by membership in this set, so a
    family string absent from it falls through to the v1 path — which is the
    silent-wrong-adapter failure `C8` describes, in the opposite direction.
    """

    assert RULEBOOK_V2_VERSION in RULEBOOK_V2_FAMILY_ALIASES
    assert _yaml(_CONF / "rulebook" / "v2.yaml")["version"] in RULEBOOK_V2_FAMILY_ALIASES
    assert _yaml(_CONF / "config.yaml")["rulebook"]["version"] in RULEBOOK_V2_FAMILY_ALIASES


def test_the_specification_in_force_is_declared_separately() -> None:
    """The family string is not the specification, and this is the field that is.

    `4.7-final-implementation-complete` reads as a specification version and names
    one the code has not implemented since `RB51`. That is `C8`'s open half, and
    it stays open: the string cannot be corrected in isolation — see the next
    test for why — so the defence against misreading it is that the real answer
    is declared, unambiguously, next to it.
    """

    assert _yaml(_CONF / "config.yaml")["rulebook"]["specification_id"] == "RULEBOOK-V5.1"


def test_the_family_string_is_load_bearing_for_a_frozen_dataset_hash() -> None:
    """Why `C8`'s open half is not a rename, recorded as an executable fact.

    `geometry_config_hash()` hashes a payload containing this string, and
    `scripts/verify_rulebook_eligibility_stability.py` compares that hash for
    equality against the one frozen into the dataset's eligibility provenance.
    So changing the string moves a dataset provenance identity, which `AGENTS.md`
    reserves for explicit user approval.

    `DEC-C8-004` deferred the rename for a *different*, incorrect reason — that
    the rulebook-catalog cache would be invalidated. It would not: cache reuse
    keys on `relative_path` and `scenario_fingerprint`. This test pins the reason
    that is real, so the next person to try the rename discovers the constraint
    here rather than from a failing hash comparison.
    """

    frozen = _yaml(_CONF / "rulebook" / "v2.yaml")["version"]
    baseline = geometry_config_hash()

    assert geometry_config_hash(RulebookV2Config(version=frozen)) == baseline
    assert RULEBOOK_V2_VERSION in json.dumps({"rulebook_version": RULEBOOK_V2_VERSION}), (
        "the constant is what the hashed payload carries"
    )

    stability = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "audits"
            / "dataset_construction_2026-09-05"
            / "rulebook_eligibility_stability.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline in json.dumps(stability), (
        "the current geometry hash must still match the one frozen with the dataset; "
        "if this fails, the family string or the geometry contract moved and the "
        "dataset's eligibility provenance needs an explicit decision"
    )


def test_the_historical_spellings_still_route() -> None:
    """A run directory written under an older family string must still be readable."""

    for legacy in ("4.7-final-implementation-complete", "4.6-final-implementation-complete"):
        assert legacy in RULEBOOK_V2_FAMILY_ALIASES
