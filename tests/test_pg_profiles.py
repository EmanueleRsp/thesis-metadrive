from __future__ import annotations

import pytest

from thesis_rl.scenarios.pg.profiles import PG_PROFILES, get_pg_profile, validate_native_tokens


def test_all_v1_profiles_are_deterministic_and_use_expected_density_ranges() -> None:
    for profile in PG_PROFILES:
        first = profile.sample(seed=123)
        second = profile.sample(seed=123)
        assert first == second
        assert profile.traffic_density_min <= first.traffic_density <= profile.traffic_density_max
        assert len(first.block_sequence) + 1 in profile.num_blocks_choices
        assert all(isinstance(token, str) for token in first.block_sequence)
        assert first.accident_prob == 0.0


def test_complex_profiles_contain_required_native_blocks() -> None:
    assert any(token in {"y", "r", "R", "O"} for token in get_pg_profile("P2_merge_or_roundabout").sample(seed=1).block_sequence)
    assert any(token in {"X", "T"} for token in get_pg_profile("P3_intersection").sample(seed=1).block_sequence)
    p5 = get_pg_profile("P5_complex_mixed").sample(seed=1)
    assert len(set(p5.block_sequence).intersection({"y", "r", "R", "O", "X", "T"})) >= 2


def test_profile_rejects_missing_native_token() -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_native_tokens(get_pg_profile("P3_intersection"), ["S", "C"])
