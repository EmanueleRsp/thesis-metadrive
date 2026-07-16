"""Small helpers for assembling thesis-specific SB3 specs.

This module is the intended landing zone for Hydra-to-SB3 translation logic.
The first concrete use is to keep SB3 planner wrappers thin while centralizing:

- policy / policy_kwargs normalization
- optional custom replay-buffer hooks
- future algorithm-specific bridge configuration
"""

from __future__ import annotations

from typing import Any

from thesis_rl.sb3_extensions.specs import Sb3AlgorithmSpec, Sb3PolicySpec


def normalize_policy_kwargs(raw_policy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize planner-provided SB3 policy kwargs into plain Python objects."""

    if raw_policy_kwargs is None:
        return {}

    policy_kwargs = dict(raw_policy_kwargs)
    net_arch = policy_kwargs.get("net_arch")
    if isinstance(net_arch, (list, tuple)):
        policy_kwargs["net_arch"] = list(net_arch)
    elif isinstance(net_arch, dict):
        policy_kwargs["net_arch"] = {
            str(key): list(value) if isinstance(value, (list, tuple)) else value
            for key, value in net_arch.items()
        }
    return policy_kwargs


def uses_explicit_custom_sb3_policy(planner_cfg: dict[str, Any]) -> bool:
    """Whether planner config already opts into a non-default SB3 policy path."""

    policy = planner_cfg.get("policy", "MlpPolicy")
    if not isinstance(policy, str):
        return True

    policy_kwargs = normalize_policy_kwargs(planner_cfg.get("policy_kwargs"))
    return "features_extractor_class" in policy_kwargs


def _activation_class(name: str):
    key = str(name).strip().lower()
    if key == "relu":
        from torch.nn import ReLU

        return ReLU
    if key == "tanh":
        from torch.nn import Tanh

        return Tanh
    if key == "gelu":
        from torch.nn import GELU

        return GELU
    raise ValueError(f"Unsupported SB3 decoder activation bridge: {name}")


def _decoder_policy_kwargs(
    backend_name: str,
    decoder_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not decoder_cfg:
        return {}

    decoder_type = str(decoder_cfg.get("type", "mlp")).strip().lower()
    if decoder_type != "mlp":
        raise ValueError(
            f"`{backend_name}` currently supports only MLP decoder bridge "
            f"configs, got decoder type '{decoder_type}'."
        )

    dropout = float(decoder_cfg.get("dropout", 0.0))
    layer_norm = bool(decoder_cfg.get("layer_norm", False))
    if dropout != 0.0:
        raise ValueError(
            f"`{backend_name}` SB3 bridge does not yet support decoder dropout, "
            f"got dropout={dropout}."
        )
    if layer_norm:
        raise ValueError(
            f"`{backend_name}` SB3 bridge does not yet support decoder layer_norm=true."
        )

    hidden_layers = [int(width) for width in decoder_cfg.get("hidden_layers", [])]
    activation_fn = _activation_class(str(decoder_cfg.get("activation", "relu")))
    is_ppo_backend = str(backend_name).lower() == "ppo_sb3"

    if is_ppo_backend:
        net_arch: dict[str, list[int]] | list[int] = {
            "pi": list(hidden_layers),
            "vf": list(hidden_layers),
        }
    else:
        net_arch = list(hidden_layers)

    return {
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }


def _should_apply_decoder_bridge(
    backend_name: str,
    decoder_cfg: dict[str, Any] | None,
) -> bool:
    """Whether decoder config should actively override planner policy kwargs.

    For canonical fork-backed baseline presets like `td3_sb3`, `sac_sb3`, and
    `ppo_sb3`, the algorithm config already carries the intended SB3-side
    `policy_kwargs`. In that case, the same-name decoder is treated as a
    compatibility/profile marker, not as an override source. Thesis-specific
    decoders such as `mlp_encoded` still apply through the bridge.
    """

    decoder_name = str((decoder_cfg or {}).get("name", "")).strip().lower()
    return bool(decoder_name) and decoder_name != str(backend_name).strip().lower()


def _maybe_encoder_policy_kwargs(
    encoder_cfg: dict[str, Any] | None,
    obs_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    encoder_type = str((encoder_cfg or {}).get("type", "none")).strip().lower()
    if encoder_type == "none":
        return {}

    try:
        from thesis_rl.sb3_extensions.features_extractors import ThesisEncoderFeatureExtractor
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on SB3 install
        raise ImportError(
            "The SB3 encoder bridge requires `stable-baselines3` to be installed. "
            "Run `uv sync` in the project environment."
        ) from exc

    return {
        "features_extractor_class": ThesisEncoderFeatureExtractor,
        "features_extractor_kwargs": {
            "cfg_encoder": dict(encoder_cfg or {}),
            "cfg_obs": {} if obs_cfg is None else dict(obs_cfg),
        },
    }


def validate_sb3_bridge_configs(
    backend_name: str,
    planner_cfg: dict[str, Any],
    *,
    encoder_cfg: dict[str, Any] | None = None,
    decoder_cfg: dict[str, Any] | None = None,
    obs_cfg: dict[str, Any] | None = None,
) -> None:
    """Validate that the current SB3 bridge configuration is intentional.

    The baseline fork-backed path currently supports the plain SB3 MLP policy
    profile directly. Thesis-specific encoder/decoder modules are expected to
    enter through an explicit custom SB3 policy/feature-extractor bridge, not
    to be ignored silently by the backend wrappers.
    """

    if uses_explicit_custom_sb3_policy(planner_cfg):
        return

    encoder_type = str((encoder_cfg or {}).get("type", "none")).strip().lower()

    errors: list[str] = []
    try:
        _decoder_policy_kwargs(backend_name, decoder_cfg)
    except ValueError as exc:
        errors.append(str(exc))
    if encoder_type != "none" and obs_cfg is None:
        errors.append(
            f"encoder type '{encoder_type}' requires observation config for the "
            "SB3 feature-extractor bridge"
        )
    if encoder_type not in {"none", "mlp", "lq"}:
        errors.append(
            f"encoder type '{encoder_type}' is not supported by the current SB3 "
            "feature-extractor bridge"
        )

    if not errors:
        return

    error_text = "; ".join(errors)
    raise ValueError(
        f"`{backend_name}` SB3 bridge configuration is invalid: {error_text}. "
        "Use a supported MLP decoder profile with encoder types supported by "
        "`thesis_rl.sb3_extensions`, or provide an explicit custom SB3 "
        "policy/feature extractor."
    )


def build_policy_spec(
    *,
    policy: str | type[Any] = "MlpPolicy",
    policy_kwargs: dict[str, Any] | None = None,
) -> Sb3PolicySpec:
    """Construct a normalized SB3 policy spec."""

    return Sb3PolicySpec(
        policy=policy,
        policy_kwargs=normalize_policy_kwargs(policy_kwargs),
    )


def build_algorithm_spec(
    *,
    replay_buffer_class: type[Any] | None = None,
    replay_buffer_kwargs: dict[str, Any] | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
) -> Sb3AlgorithmSpec:
    """Construct a normalized SB3 algorithm-side spec."""

    return Sb3AlgorithmSpec(
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs={} if replay_buffer_kwargs is None else dict(replay_buffer_kwargs),
        algorithm_kwargs={} if algorithm_kwargs is None else dict(algorithm_kwargs),
    )


def build_policy_spec_from_planner_cfg(
    planner_cfg: dict[str, Any],
    *,
    default_policy: str = "MlpPolicy",
) -> Sb3PolicySpec:
    """Build a normalized policy spec from planner config payload."""

    return build_policy_spec(
        policy=planner_cfg.get("policy", default_policy),
        policy_kwargs=planner_cfg.get("policy_kwargs"),
    )


def build_algorithm_spec_from_planner_cfg(planner_cfg: dict[str, Any]) -> Sb3AlgorithmSpec:
    """Build algorithm-side customization hooks from planner config payload.

    The current baseline path keeps these optional. They exist now so the forked
    SB3 integration can grow support for thesis-specific replay buffers without
    expanding planner wrapper responsibilities again.
    """

    return build_algorithm_spec(
        replay_buffer_class=planner_cfg.get("replay_buffer_class"),
        replay_buffer_kwargs=planner_cfg.get("replay_buffer_kwargs"),
        algorithm_kwargs=planner_cfg.get("algorithm_kwargs"),
    )


def build_sb3_specs_from_configs(
    backend_name: str,
    planner_cfg: dict[str, Any],
    *,
    encoder_cfg: dict[str, Any] | None = None,
    decoder_cfg: dict[str, Any] | None = None,
    obs_cfg: dict[str, Any] | None = None,
    default_policy: str = "MlpPolicy",
) -> tuple[Sb3PolicySpec, Sb3AlgorithmSpec]:
    """Build the current SB3 bridge specs from planner/network configs."""

    validate_sb3_bridge_configs(
        backend_name=backend_name,
        planner_cfg=planner_cfg,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        obs_cfg=obs_cfg,
    )
    policy_spec = build_policy_spec_from_planner_cfg(
        planner_cfg,
        default_policy=default_policy,
    )
    algorithm_spec = build_algorithm_spec_from_planner_cfg(planner_cfg)
    if uses_explicit_custom_sb3_policy(planner_cfg):
        return policy_spec, algorithm_spec

    bridged_policy_kwargs = dict(policy_spec.policy_kwargs)
    if _should_apply_decoder_bridge(backend_name, decoder_cfg):
        bridged_policy_kwargs.update(_decoder_policy_kwargs(backend_name, decoder_cfg))
    bridged_policy_kwargs.update(_maybe_encoder_policy_kwargs(encoder_cfg, obs_cfg))
    if str((encoder_cfg or {}).get("type", "none")).strip().lower() != "none":
        if str(backend_name).lower() in {"td3_sb3", "sac_sb3"}:
            bridged_policy_kwargs["share_features_extractor"] = False
        elif str(backend_name).lower() == "ppo_sb3":
            bridged_policy_kwargs["share_features_extractor"] = True
            bridged_policy_kwargs["ortho_init"] = False
    return build_policy_spec(
        policy=policy_spec.policy,
        policy_kwargs=bridged_policy_kwargs,
    ), algorithm_spec
