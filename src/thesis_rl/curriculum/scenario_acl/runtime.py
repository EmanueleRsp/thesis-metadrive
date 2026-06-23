from __future__ import annotations

from omegaconf import DictConfig

from thesis_rl.curriculum.config import CurriculumConfig


_SUPPORTED_REWARD_BEHAVIORS = {
    "off",
    "monitor_only",
    "scalar_reward",
    "hybrid",
    "lexicographic",
}


def validate_scenario_acl_runtime_support(
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    *,
    context: str,
) -> None:
    """Fail fast for runtime combinations not yet implemented for scenario ACL."""
    if not curriculum_cfg.is_scenario_acl:
        return

    vectorized_cfg = cfg.env.get("vectorized", {})
    if bool(vectorized_cfg.get("enabled", False)):
        raise ValueError(
            "Curriculum kind 'scenario_acl' currently requires "
            "env.vectorized.enabled=false."
        )

    reward_behavior = str(cfg.reward.get("behavior", "")).strip().lower()
    if reward_behavior not in _SUPPORTED_REWARD_BEHAVIORS:
        supported = ", ".join(sorted(_SUPPORTED_REWARD_BEHAVIORS))
        raise ValueError(
            "Curriculum kind 'scenario_acl' does not support "
            f"reward.behavior='{cfg.reward.get('behavior')}' in {context}. "
            f"Supported values: {supported}."
        )

    if bool(curriculum_cfg.scenario_acl.use_mutation):
        raise ValueError(
            "Curriculum kind 'scenario_acl' does not yet support "
            "scenario_acl.use_mutation=true. Use replay-only mode for now."
        )
