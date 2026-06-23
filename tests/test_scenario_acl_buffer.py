from __future__ import annotations

import numpy as np

from thesis_rl.curriculum import (
    ScenarioAclReplaySamplingConfig,
    ScenarioBuffer,
    ScenarioRecord,
    compute_replay_probabilities,
)


def _record(
    scenario_id: str,
    *,
    usefulness: float,
    scenario_hash: str,
    last_seen_step: int = 0,
) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id=scenario_id,
        source="generate",
        parent_id=None,
        scenario_description_path=f"/tmp/{scenario_id}.pkl",
        scenario_description_hash=scenario_hash,
        dataset_directory="/tmp",
        scenario_index=0,
        env_config={"start_seed": 0},
        reset_seed=0,
        generator_arm="broad_random",
        mutation_type=None,
        mutation_params=None,
        validation_status="valid",
        rule_criticality=0.0,
        learning_potential=usefulness,
        usefulness=usefulness,
        usefulness_norm=1.0,
        rank=0,
        num_seen=1,
        last_seen_step=last_seen_step,
        num_children=0,
        metrics_summary={},
    )


def test_scenario_buffer_rejects_duplicate_hashes() -> None:
    buffer = ScenarioBuffer(capacity=2)

    assert buffer.insert(_record("a", usefulness=1.0, scenario_hash="same")) is True
    assert buffer.insert(_record("b", usefulness=2.0, scenario_hash="same")) is False
    assert len(buffer) == 1


def test_scenario_buffer_replaces_worst_when_capacity_is_full() -> None:
    buffer = ScenarioBuffer(capacity=2)
    buffer.insert(_record("a", usefulness=1.0, scenario_hash="a"))
    buffer.insert(_record("b", usefulness=2.0, scenario_hash="b"))

    assert buffer.insert(_record("c", usefulness=3.0, scenario_hash="c")) is True

    records = buffer.records()
    assert {record.scenario_id for record in records} == {"b", "c"}
    assert sorted(record.rank for record in records) == [1, 2]


def test_compute_replay_probabilities_mix_usefulness_and_staleness() -> None:
    records = [
        _record("best", usefulness=4.0, scenario_hash="a", last_seen_step=9),
        _record("mid", usefulness=2.0, scenario_hash="b", last_seen_step=2),
    ]
    records[0].rank = 1
    records[1].rank = 2

    probabilities = compute_replay_probabilities(
        records,
        current_step=10,
        cfg=ScenarioAclReplaySamplingConfig(omega=0.5, beta=1.0, staleness_offset=1),
        use_staleness=True,
    )

    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[1] > 0.0
    assert probabilities[1] > probabilities[0]


def test_scenario_buffer_sample_replay_returns_probabilities() -> None:
    rng = np.random.default_rng(7)
    buffer = ScenarioBuffer(capacity=3)
    buffer.insert(_record("a", usefulness=3.0, scenario_hash="a", last_seen_step=5))
    buffer.insert(_record("b", usefulness=2.0, scenario_hash="b", last_seen_step=1))

    selection = buffer.sample_replay(
        rng=rng,
        current_step=6,
        cfg=ScenarioAclReplaySamplingConfig(),
        use_staleness=True,
    )

    assert selection.record.scenario_id in {"a", "b"}
    assert len(selection.probabilities) == 2
    assert np.isclose(sum(selection.probabilities), 1.0)
