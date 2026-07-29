from __future__ import annotations

import pytest

from thesis_rl.scenarios.pg.generator import _realized_static_obstacle_metadata
from thesis_rl.scenarios.pg.profiles import (
    PG_PROFILES,
    PGProfile,
    get_pg_profile,
    validate_native_tokens,
)


EXPECTED_ACCIDENT_PROBABILITIES = {
    "P0_simple": 0.00,
    "P1_vehicle_interaction": 0.03,
    "P2_merge_or_roundabout": 0.08,
    "P3_intersection": 0.08,
    "P5_complex_mixed": 0.15,
}


def test_all_v1_profiles_are_deterministic_and_use_expected_density_ranges() -> None:
    for profile in PG_PROFILES:
        first = profile.sample(seed=123)
        second = profile.sample(seed=123)
        assert first == second
        assert profile.traffic_density_min <= first.traffic_density <= profile.traffic_density_max
        assert len(first.block_sequence) + 1 in profile.num_blocks_choices
        assert all(isinstance(token, str) for token in first.block_sequence)
        assert first.accident_prob == EXPECTED_ACCIDENT_PROBABILITIES[profile.name]


def test_complex_profiles_contain_required_native_blocks() -> None:
    assert any(token in {"y", "r", "R", "O"} for token in get_pg_profile("P2_merge_or_roundabout").sample(seed=1).block_sequence)
    assert any(token in {"X", "T"} for token in get_pg_profile("P3_intersection").sample(seed=1).block_sequence)
    p5 = get_pg_profile("P5_complex_mixed").sample(seed=1)
    assert len(set(p5.block_sequence).intersection({"y", "r", "R", "O", "X", "T"})) >= 2


def test_profile_rejects_missing_native_token() -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_native_tokens(get_pg_profile("P3_intersection"), ["S", "C"])


def test_profile_rejects_an_invalid_accident_probability() -> None:
    with pytest.raises(ValueError, match="within \[0, 1\]"):
        PGProfile("P0_simple", ("S",), 0.0, 0.0, (2,), accident_prob=1.01)


def test_realized_static_obstacle_metadata_ignores_non_accident_objects() -> None:
    class TrafficBarrier:
        pass

    class Vehicle:
        pass

    class _Engine:
        @staticmethod
        def get_objects():
            return {"barrier": TrafficBarrier(), "vehicle": Vehicle()}

    class _Env:
        engine = _Engine()

    assert _realized_static_obstacle_metadata(_Env()) == {
        "realized": True,
        "object_types": ["TrafficBarrier"],
    }
