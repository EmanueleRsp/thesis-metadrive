"""Sub-rule dominance and cost diagnostics (EP-SUBRULE-DIAG).

Requirement IDs refer to
`docs/implementation/subrule_dominance_diagnostics_exec_plan.md`.
"""

from __future__ import annotations

from thesis_rl.rulebook.v2.subrule_diagnostics import (
    SubruleEpisodeAccumulator,
    aggregate_subrule_episodes,
    extract_subrule_step,
)


def _component(name: str, cost: float, *, applicable: bool = True) -> dict:
    return {
        "name": name,
        "cost": cost,
        "raw": {},
        "applicable": applicable,
        "evaluable": True,
        "status": "violated" if cost > 0.0 else "satisfied",
        "diagnostics": {},
    }


def _macro_step(
    *,
    rss: float,
    ttc: float,
    applicable: bool = True,
) -> dict:
    """One step with R2 = max(rss, ttc); R1/R3 absent (out of scope, DEC-SUB-002)."""

    subcomponents = (
        _component("rss", rss, applicable=applicable),
        _component("ttc", ttc, applicable=applicable),
    )
    worst = max(subcomponents, key=lambda c: (c["cost"], c["name"]))
    macro_cost = worst["cost"] if applicable else 0.0
    return {
        "rule_components": {
            "interaction_risk": {
                "name": "interaction_risk",
                "cost": macro_cost,
                "raw": {"worst_component": worst["name"], "subcomponents": subcomponents},
                "applicable": applicable,
                "evaluable": True,
                "status": "violated" if macro_cost > 0.0 else "satisfied",
                "diagnostics": {"worst_component": worst["name"]},
            }
        }
    }


# --- extract_subrule_step ---


def test_extract_subrule_step_reads_costs_and_worst_component() -> None:
    step = _macro_step(rss=0.3, ttc=0.7)
    subrules, macros = extract_subrule_step(step)

    assert subrules["rss"].cost == 0.3
    assert subrules["ttc"].cost == 0.7
    assert subrules["rss"].macro_rule == "interaction_risk"
    macro = macros["interaction_risk"]
    assert macro.applicable is True
    assert macro.violated is True
    assert macro.worst_component == "ttc"
    assert macro.violated_subrule_count == 2  # both > 0


def test_extract_subrule_step_missing_rule_components_is_empty() -> None:
    assert extract_subrule_step({}) == ({}, {})
    assert extract_subrule_step("not a dict") == ({}, {})


def test_extract_subrule_step_not_applicable_macro_has_no_worst_component() -> None:
    step = _macro_step(rss=0.0, ttc=0.0, applicable=False)
    _subrules, macros = extract_subrule_step(step)
    macro = macros["interaction_risk"]
    assert macro.applicable is False
    assert macro.violated is False
    assert macro.worst_component is None


# --- SubruleEpisodeAccumulator: REQ-SUB-01 / REQ-SUB-02 / REQ-SUB-05 ---


def test_dominance_counts_only_violated_macro_steps() -> None:
    """AC-SUB-02: an all-zero-cost step contributes no dominance count."""

    acc = SubruleEpisodeAccumulator()
    acc.observe(_macro_step(rss=0.0, ttc=0.0))  # macro satisfied: not counted
    acc.observe(_macro_step(rss=0.9, ttc=0.1))  # macro violated: rss wins
    acc.observe(_macro_step(rss=0.2, ttc=0.8))  # macro violated: ttc wins
    summary = acc.finalize()

    assert summary["rss"]["worst_component_count"] == 1
    assert summary["ttc"]["worst_component_count"] == 1
    assert summary["rss"]["macro_violated_step_count"] == 2
    # 3 steps applicable in total for each sub-rule (all 3 macro-applicable steps)
    assert summary["rss"]["applicable_step_count"] == 3
    assert summary["rss"]["total_step_count"] == 3


def test_multi_violation_share_fixture() -> None:
    """AC-SUB-05: matches a hand-computed fixture."""

    acc = SubruleEpisodeAccumulator()
    acc.observe(_macro_step(rss=0.5, ttc=0.5))  # both violated: multi
    acc.observe(_macro_step(rss=0.5, ttc=0.0))  # only rss violated: not multi
    acc.observe(_macro_step(rss=0.0, ttc=0.0))  # macro satisfied
    summary = acc.finalize()

    assert summary["rss"]["macro_violated_step_count"] == 2
    assert summary["rss"]["macro_multi_violation_step_count"] == 1


def test_not_applicable_subrule_step_excluded_from_stats() -> None:
    acc = SubruleEpisodeAccumulator()
    step = _macro_step(rss=0.4, ttc=0.6)
    step["rule_components"]["interaction_risk"]["raw"]["subcomponents"][0][
        "applicable"
    ] = False
    acc.observe(step)
    summary = acc.finalize()

    assert summary["rss"]["applicable_step_count"] == 0
    assert summary["rss"]["cost_sum"] == 0.0
    # ttc still counted normally
    assert summary["ttc"]["applicable_step_count"] == 1


# --- aggregate_subrule_episodes: REQ-SUB-01 / REQ-SUB-04 ---


def test_all_episodes_contribute_and_are_disaggregated_by_source() -> None:
    """AC-SUB-01/AC-SUB-04: every episode counts, grouped by scenario source."""

    waymo_acc = SubruleEpisodeAccumulator()
    waymo_acc.observe(_macro_step(rss=0.8, ttc=0.2))
    waymo_summary = waymo_acc.finalize()

    pg_acc = SubruleEpisodeAccumulator()
    pg_acc.observe(_macro_step(rss=0.1, ttc=0.9))
    pg_summary = pg_acc.finalize()

    rows = aggregate_subrule_episodes([waymo_summary, pg_summary], ["waymo", "pg"])
    sources = {row["scenario_source"] for row in rows}
    assert sources == {"waymo", "pg"}
    waymo_rss = next(
        r for r in rows if r["scenario_source"] == "waymo" and r["subrule_name"] == "rss"
    )
    assert waymo_rss["dominance_share"] == 1.0
    assert waymo_rss["applicable_episode_count"] == 1
    assert waymo_rss["excluded_episode_count"] == 0
    pg_rss = next(r for r in rows if r["scenario_source"] == "pg" and r["subrule_name"] == "rss")
    assert pg_rss["dominance_share"] == 0.0


def test_missing_source_is_grouped_as_unknown_not_dropped() -> None:
    acc = SubruleEpisodeAccumulator()
    acc.observe(_macro_step(rss=0.5, ttc=0.5))
    rows = aggregate_subrule_episodes([acc.finalize()], [None])
    assert all(row["scenario_source"] == "unknown" for row in rows)


def test_episode_with_no_applicable_steps_is_reported_as_never_applicable() -> None:
    """A sub-rule that is never applicable still gets a row (liveness, REQ-SUB-01),
    not a violation: it is excluded from the episode count feeding violation_rate."""

    acc = SubruleEpisodeAccumulator()
    acc.observe(_macro_step(rss=0.0, ttc=0.0, applicable=False))
    rows = aggregate_subrule_episodes([acc.finalize()], ["waymo"])
    assert rows
    for row in rows:
        assert row["applicability_rate"] == 0.0
        assert row["applicable_episode_count"] == 0
        assert row["excluded_episode_count"] == 1
        assert row["violation_rate"] == 0.0


def test_mismatched_lengths_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        aggregate_subrule_episodes([{}], ["waymo", "pg"])
