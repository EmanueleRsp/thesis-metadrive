"""Audit 2026-09-07, C35: a frozen semantic v3 observation setting cannot be ignored.

`conf/obs/semantic_v3.yaml` declares twenty keys and exactly three of them have a
reader — the signal-camera geometry ADR-045 deliberately allows overriding.
Everything else is hard-coded where the configuration cannot reach it: the token
counts and history lengths in `SemanticObservationSchemaV12`, whose `flat_dim` of
3011 `OBS-V1.3.1` freezes; the radii and the prediction horizon in the builder
kwargs of `envs/thesis_scenario_env.py`; the LiDAR geometry in the literals
`envs/factory.py` writes into `vehicle_config`.

So `obs.dynamic_radius_m=80` parsed, was logged, reached the checkpoint's
configuration record and changed nothing. This is `C24` one layer down, and it
takes `C24`'s remedy: refuse the value, keep the declaration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from thesis_rl.envs.factory import (
    _FROZEN_SEMANTIC_V3_SETTINGS,
    _configure_agent_observation,
)

_SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "obs" / "semantic_v3.yaml"


def _shipped_config() -> dict[str, object]:
    loaded = OmegaConf.load(_SHIPPED_CONFIG_PATH)
    return OmegaConf.to_container(loaded, resolve=True)  # type: ignore[return-value]


def _configure(**overrides: object) -> dict[str, object]:
    env_cfg: dict[str, object] = {}
    _configure_agent_observation(env_cfg, {**_shipped_config(), **overrides})
    return env_cfg


def test_the_shipped_configuration_still_installs_the_observation() -> None:
    """The guard must cost the production path nothing."""

    from thesis_rl.envs.observations.semantic_state_v3 import SemanticStateObservationV3

    env_cfg = _configure()

    assert env_cfg["agent_observation"] is SemanticStateObservationV3
    # The three keys ADR-045 does thread through, at the `OBS-V1.2` §6.2 baseline.
    assert env_cfg["semantic_v3_signal_range_m"] == pytest.approx(80.0)
    assert env_cfg["semantic_v3_signal_fov_degrees"] == pytest.approx(65.0)
    assert env_cfg["semantic_v3_signal_camera_height_m"] == pytest.approx(1.2)


@pytest.mark.parametrize(("key", "frozen"), list(_FROZEN_SEMANTIC_V3_SETTINGS))
def test_overriding_a_hard_coded_setting_fails_instead_of_being_ignored(
    key: str, frozen: object
) -> None:
    override: object = "allowed" if isinstance(frozen, str) else float(frozen) + 1.0
    with pytest.raises(ValueError, match=key):
        _configure(**{key: override})


def test_a_fractional_override_of_an_integer_setting_is_refused() -> None:
    """`int(5.4)` equals a frozen 5, so an int comparison would accept it.

    The guard exists to refuse silent acceptance, so the one comparison that can
    silently accept is the one it must not use.
    """

    with pytest.raises(ValueError, match="history_length"):
        _configure(history_length=5.4)
    with pytest.raises(ValueError, match="lidar_beams"):
        _configure(lidar_beams=240.9)


@pytest.mark.parametrize(
    "key", ["future_ground_truth", "other_agent_navigation", "future_disambiguation"]
)
def test_an_absent_causality_declaration_is_refused(key: str) -> None:
    """Presence, not merely agreement: an omitted declaration is not compliance.

    Nothing reads these keys, so a configuration that drops one would run with
    the contract silently unstated -- `C8`'s failure, where a missing field is
    taken for the expected value instead of being reported as absent.
    """

    config = _shipped_config()
    del config[key]

    with pytest.raises(ValueError, match=key):
        _configure_agent_observation({}, config)


def test_the_signal_camera_geometry_stays_overridable() -> None:
    """ADR-045's three keys are connected, so the guard must not cover them."""

    guarded = {key for key, _ in _FROZEN_SEMANTIC_V3_SETTINGS}
    threaded = {"signal_range_m", "signal_fov_degrees", "signal_camera_height_m"}

    assert not guarded & threaded

    env_cfg = _configure(signal_range_m=120.0)
    assert env_cfg["semantic_v3_signal_range_m"] == pytest.approx(120.0)


def test_every_shipped_key_is_either_threaded_or_guarded() -> None:
    """The guard list and the shipped config must not drift apart.

    A key added to the YAML without being either wired up or guarded is exactly
    the defect this test exists to prevent, one knob later.
    """

    guarded = {key for key, _ in _FROZEN_SEMANTIC_V3_SETTINGS}
    threaded = {"signal_range_m", "signal_fov_degrees", "signal_camera_height_m"}
    descriptive = {"name", "type"}

    unaccounted = set(_shipped_config()) - guarded - threaded - descriptive
    assert not unaccounted, (
        f"semantic v3 keys neither guarded nor passed through: {sorted(unaccounted)}"
    )


def test_no_frozen_key_reaches_the_environment_configuration() -> None:
    """Why the override was inert, stated as a property rather than as history.

    The guard is legitimate only as long as these keys influence nothing. If one
    of them ever becomes threaded, this fails and the guard has to be dropped
    instead of being left refusing a value that now works.
    """

    env_cfg = _configure()

    for key, _ in _FROZEN_SEMANTIC_V3_SETTINGS:
        assert key not in env_cfg
        assert f"semantic_v3_{key}" not in env_cfg


def test_the_frozen_values_match_what_the_shipped_config_declares() -> None:
    """Pin the guard to the file it is guarding, so a deliberate change moves both."""

    shipped = _shipped_config()
    for key, frozen in _FROZEN_SEMANTIC_V3_SETTINGS:
        assert key in shipped, f"{key} is guarded but no longer declared"
        if isinstance(frozen, str):
            assert str(shipped[key]).strip().lower() == frozen
        else:
            assert float(shipped[key]) == pytest.approx(float(frozen))
