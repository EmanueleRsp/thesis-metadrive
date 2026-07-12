from __future__ import annotations

from types import SimpleNamespace

import pytest

from thesis_rl.envs.scene_context import SceneContextAdapter
from thesis_rl.envs.thesis_scenario_env import scenario_time_limit_reached
from thesis_rl.runtime.wiring.builders import collect_scenario_runtime_stats


@pytest.mark.parametrize(
    ("steps", "length", "extra", "expected"),
    [
        (9, 10, 0, False),
        (10, 10, 0, True),
        (59, 10, 50, False),
        (60, 10, 50, True),
    ],
)
def test_scenario_time_limit_supports_zero_and_tail(
    steps: int, length: int, extra: int, expected: bool
) -> None:
    assert scenario_time_limit_reached(
        episode_steps=steps,
        scenario_length=length,
        extra_steps_after_scenario=extra,
    ) is expected


def test_scene_context_separates_line_from_physical_boundary() -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace(
        on_yellow_continuous_line=True,
        on_white_continuous_line=False,
        crash_sidewalk=False,
        navigation=SimpleNamespace(current_lateral=1.0, route_completion=0.4),
        LENGTH=4.5,
        WIDTH=1.9,
    )
    env = SimpleNamespace(config={"max_lateral_dist": 4.0})

    assert adapter.is_on_continuous_line(vehicle) is True
    assert adapter.is_physically_out_of_road(env, vehicle) is False
    assert adapter.get_ego_dimensions(vehicle) == (4.5, 1.9)

    vehicle.navigation.current_lateral = 5.0
    assert adapter.is_physically_out_of_road(env, vehicle) is True


def test_scene_context_termination_reason_is_stable() -> None:
    adapter = SceneContextAdapter()
    vehicle = SimpleNamespace()
    assert adapter.get_termination_reason(
        None,
        vehicle,
        {"crash_vehicle": True, "max_step": True},
    ) == "crash_vehicle"


def test_scenario_time_limit_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        scenario_time_limit_reached(
            episode_steps=0,
            scenario_length=0,
            extra_steps_after_scenario=0,
        )


def test_collect_scenario_runtime_stats_merges_vector_workers() -> None:
    class FakeVectorEnv:
        def env_method(self, name: str):
            assert name == "get_runtime_stats"
            return [
                {
                    "resets": 2,
                    "steps": 4,
                    "episodes": 1,
                    "resets_by_source": {"waymo": 2},
                    "steps_by_source": {"waymo": 4},
                    "episodes_by_arm": {"A0": 1},
                    "termination_reasons": {"success": 1},
                },
                {
                    "resets": 1,
                    "steps": 3,
                    "episodes": 1,
                    "resets_by_source": {"pg": 1},
                    "steps_by_source": {"pg": 3},
                    "episodes_by_arm": {"A1": 1},
                    "termination_reasons": {"truncated": 1},
                },
            ]

    stats = collect_scenario_runtime_stats(FakeVectorEnv())
    assert stats is not None
    assert stats["resets"] == 3
    assert stats["steps_by_source"] == {"waymo": 4, "pg": 3}
    assert stats["episodes_by_arm"] == {"A0": 1, "A1": 1}
