from __future__ import annotations

from thesis_rl.scenarios.pg.replenishment import plan_profile_counts
from thesis_rl.scenarios.pg.report import _resolve_profile_counts


def test_profile_override_disables_unspecified_profiles() -> None:
    counts = _resolve_profile_counts(350, {"P5_complex_mixed": 1750})

    assert counts["P5_complex_mixed"] == 1750
    assert sum(counts.values()) == 1750
    assert all(counts[profile] == 0 for profile in counts if profile != "P5_complex_mixed")


def test_targeted_replenishment_uses_profiles_for_missing_complex_arms() -> None:
    report = {
        "runtime_eligible_by_arm": {
            "A0_simple_low_traffic": 1876,
            "A1_traffic": 798,
            "A2_junction": 177,
            "A3_complex_junction": 24,
            "A5_critical_mixed": 18,
        },
        "selection_diagnostics": {
            split: {
                "arms": {
                    "A0_simple_low_traffic": {"sources": {"pg": {"target": 572 // 4}}},
                    "A1_traffic": {"sources": {"pg": {"target": 301 // 4}}},
                    "A2_junction": {"sources": {"pg": {"target": 251 // 4}}},
                    "A3_complex_junction": {"sources": {"pg": {"target": 29 // 4}}},
                    "A5_critical_mixed": {"sources": {"pg": {"target": 25 // 4}}},
                }
            }
            for split in ("train", "validation", "test")
        },
    }

    counts = plan_profile_counts(report, budget=1750)

    assert sum(counts.values()) == 1750
    assert set(counts) == {
        "P2_merge_or_roundabout",
        "P3_intersection",
        "P5_complex_mixed",
    }


def test_targeted_replenishment_returns_empty_when_no_arm_is_missing() -> None:
    report = {
        "runtime_eligible_by_arm": {"A0_simple_low_traffic": 100},
        "selection_diagnostics": {
            "train": {"arms": {"A0_simple_low_traffic": {"sources": {"pg": {"target": 10}}}}}
        },
    }

    assert plan_profile_counts(report) == {}


def test_targeted_replenishment_uses_global_source_capacity() -> None:
    report = {
        "runtime_eligible_by_source_arm": {
            "pg": {"A5_critical_mixed": 25, "A4_vru": 0},
            "waymo": {"A5_critical_mixed": 276, "A4_vru": 328},
        },
        "selection_diagnostics": {
            split: {
                "arms": {
                    "A5_critical_mixed": {"target": 583},
                    "A4_vru": {"target": 583},
                }
            }
            for split in ("train", "validation", "test")
        },
    }

    counts = plan_profile_counts(report, budget=1750)

    assert counts == {"P5_complex_mixed": 1750}
