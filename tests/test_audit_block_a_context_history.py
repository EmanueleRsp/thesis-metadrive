"""Audit 2026-09-06, block A5: a repeated ``build`` for one step must not
consume a second slot of the 21-step compliance-trace window.

MetaDrive builds the observation before the Rulebook wrapper publishes the
committed context, and the wrapper rebuilds it afterwards, so every step is
traced twice. Appending both rows halved the effective ``context_history``
window from 21 steps to about 10.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import thesis_rl.envs.observations.causal_semantic as causal_semantic
from test_causal_semantic_batch import _Vehicle, _actor, _context, _route
from thesis_rl.envs.observations.causal_semantic import PerceptionBoundedSemanticBatchBuilder


def _builder(monkeypatch):
    route, lanes = _route()
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    return PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0), (
        route,
        lanes,
    )


def test_double_build_per_step_keeps_the_full_history_window(monkeypatch) -> None:
    builder, (route, lanes) = _builder(monkeypatch)
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))

    batch = None
    for step in range(30):
        context = _context(step, ego, (), route, lanes)
        builder.build(_Vehicle(), context)  # stale build (MetaDrive)
        batch = builder.build(_Vehicle(), context)  # committed build (wrapper)

    assert batch is not None
    assert batch.context_history_mask.tolist() == [1.0] * 21
    assert len(builder._context_rows) == 21
    assert [step for step, _row in builder._context_rows] == list(range(9, 30))


def test_repeated_build_reuses_the_committed_row(monkeypatch) -> None:
    builder, (route, lanes) = _builder(monkeypatch)
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))

    first = builder.build(_Vehicle(), _context(0, ego, (), route, lanes))
    faster_ego = _actor("ego", (0.0, 0.0), (10.0, 0.0))
    second = builder.build(_Vehicle(), _context(0, faster_ego, (), route, lanes))

    assert np.array_equal(first.context_history, second.context_history)
    assert second.context_history_mask.tolist() == [0.0] * 20 + [1.0]
