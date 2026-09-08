"""REQ-AF-01 (OBS-AUDIT-FIX-001): the LiDAR admission sweep must publish the
same actor identity as the Rulebook's live ``ActorSnapshot.actor_id``.

ScenarioNet replay names every MetaDrive object with a random string unless
``force_reuse_object_name`` is set, while ``metadrive_live._stable_actor_id``
maps that name to the source actor id.  Before the fix the sweep returned the
raw object names, the two sets were disjoint and no actor was ever admitted to
the semantic tracker.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import metadrive.component.sensors.distance_detector as distance_detector
import metadrive.utils.utils as metadrive_utils

from thesis_rl.envs.observations.perception import FirstHitLidarAdapter


class _Hit:
    def __init__(self, obj: object) -> None:
        self._obj = obj

    def hasHit(self) -> bool:  # noqa: N802 - Bullet API spelling
        return True

    def getNode(self) -> object:  # noqa: N802 - Bullet API spelling
        return self._obj


def _install_fake_sweep(monkeypatch, objects: list[object]) -> None:
    result = SimpleNamespace(
        cloud_points=np.ones(240, dtype=np.float32),
        detected_objects=[_Hit(obj) for obj in objects],
    )
    monkeypatch.setattr(
        distance_detector.DistanceDetector, "perceive", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(metadrive_utils, "get_object_from_node", lambda node: node)


def _vehicle(mapping: dict[str, str] | None) -> SimpleNamespace:
    traffic_manager = SimpleNamespace()
    if mapping is not None:
        traffic_manager.obj_id_to_scenario_id = mapping
    engine = SimpleNamespace(
        traffic_manager=traffic_manager,
        get_sensor=lambda name: object(),
        physics_world=SimpleNamespace(dynamic_world=object()),
    )
    return SimpleNamespace(id="ego-object", engine=engine)


def test_sweep_resolves_scenario_ids_through_traffic_manager_mapping(monkeypatch) -> None:
    other = SimpleNamespace(id="3c356926-random-object-name")
    ego = _vehicle({"3c356926-random-object-name": "77", "ego-object": "ego"})
    _install_fake_sweep(monkeypatch, [other, ego])

    sweep = FirstHitLidarAdapter().sweep(ego)

    assert sweep.actor_ids == frozenset({"77"})
    assert sweep.hit_count == 2


def test_sweep_keeps_object_ids_without_scenario_mapping(monkeypatch) -> None:
    barrier = SimpleNamespace(id="barrier-object")
    ego = _vehicle(None)
    _install_fake_sweep(monkeypatch, [barrier, ego])

    sweep = FirstHitLidarAdapter().sweep(ego)

    assert sweep.actor_ids == frozenset({"barrier-object"})


def test_sweep_keeps_unmapped_objects_alongside_mapped_actors(monkeypatch) -> None:
    replayed = SimpleNamespace(id="random-a")
    cone = SimpleNamespace(id="cone-object")
    ego = _vehicle({"random-a": "12"})
    _install_fake_sweep(monkeypatch, [replayed, cone])

    sweep = FirstHitLidarAdapter().sweep(ego)

    assert sweep.actor_ids == frozenset({"12", "cone-object"})


@pytest.mark.parametrize("beams", [239, 241])
def test_sweep_still_rejects_wrong_beam_count(monkeypatch, beams: int) -> None:
    result = SimpleNamespace(cloud_points=np.ones(beams, dtype=np.float32), detected_objects=[])
    monkeypatch.setattr(
        distance_detector.DistanceDetector, "perceive", lambda *args, **kwargs: result
    )
    with pytest.raises(RuntimeError, match="expected 240"):
        FirstHitLidarAdapter().sweep(_vehicle(None))
