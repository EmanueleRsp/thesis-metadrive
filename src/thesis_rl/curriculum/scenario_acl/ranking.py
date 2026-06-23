from __future__ import annotations

from typing import Sequence

import numpy as np

from thesis_rl.curriculum.config import ScenarioAclReplaySamplingConfig
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord


def assign_ranks(records: Sequence[ScenarioRecord]) -> list[ScenarioRecord]:
    ranked = sorted(records, key=lambda record: record.usefulness, reverse=True)
    for idx, record in enumerate(ranked, start=1):
        record.rank = idx
    return ranked


def compute_replay_probabilities(
    records: Sequence[ScenarioRecord],
    *,
    current_step: int,
    cfg: ScenarioAclReplaySamplingConfig,
    use_staleness: bool,
) -> np.ndarray:
    if not records:
        raise ValueError("Cannot compute replay probabilities for an empty buffer.")

    ranks = np.asarray([max(int(record.rank), 1) for record in records], dtype=np.float64)
    p_usefulness = np.power(ranks, -float(cfg.beta))
    p_usefulness = p_usefulness / p_usefulness.sum()

    if use_staleness:
        staleness = np.asarray(
            [
                max(
                    float(cfg.staleness_offset),
                    float(cfg.staleness_offset + current_step - int(record.last_seen_step)),
                )
                for record in records
            ],
            dtype=np.float64,
        )
        p_staleness = staleness / staleness.sum()
    else:
        p_staleness = np.full_like(p_usefulness, 1.0 / len(records))

    omega = float(cfg.omega)
    probabilities = (omega * p_usefulness) + ((1.0 - omega) * p_staleness)
    return probabilities / probabilities.sum()
