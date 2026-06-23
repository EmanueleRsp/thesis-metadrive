from __future__ import annotations

import numpy as np

from thesis_rl.envs.observations.semantic_state import SemanticStateObservation


class _DummyNavigation:
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints


class _DummyVehicle:
    def __init__(self, checkpoints):
        self.navigation = _DummyNavigation(checkpoints)


def test_route_pairs_accept_numpy_checkpoints() -> None:
    vehicle = _DummyVehicle(
        checkpoints=[
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([2.0, 1.0], dtype=np.float32),
        ]
    )

    route_pairs = SemanticStateObservation._route_pairs(vehicle)

    assert route_pairs == {
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (2.0, 1.0)),
    }


def test_is_on_route_normalizes_numpy_lane_tokens() -> None:
    route_pairs = {
        ((0.0, 0.0), (1.0, 0.0)),
    }

    assert (
        SemanticStateObservation._is_on_route(
            (
                np.asarray([0.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 0.0], dtype=np.float32),
                0,
            ),
            route_pairs,
        )
        is True
    )
