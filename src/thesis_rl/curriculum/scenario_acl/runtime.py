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
        num_envs = int(vectorized_cfg.get("num_envs", 1))
        start_method = str(vectorized_cfg.get("start_method", "spawn")).lower()
        if num_envs <= 1:
            raise ValueError("ACL vector execution requires env.vectorized.num_envs > 1.")
        if start_method != "spawn":
            raise ValueError(
                "Scenario ACL vector execution requires env.vectorized.start_method='spawn'."
            )
        if int(curriculum_cfg.scenario_acl.mab.num_arms) != 6:
            raise ValueError("ScenarioNet ACL vector execution requires exactly six A0-A5 arms.")

    env_name = str(cfg.env.get("name", "")).strip().lower()
    if env_name != "scenarionet":
        raise ValueError("scenario_acl requires env=scenarionet.")

    reward_behavior = str(cfg.reward.get("behavior", "")).strip().lower()
    if reward_behavior not in _SUPPORTED_REWARD_BEHAVIORS:
        supported = ", ".join(sorted(_SUPPORTED_REWARD_BEHAVIORS))
        raise ValueError(
            "Curriculum kind 'scenario_acl' does not support "
            f"reward.behavior='{cfg.reward.get('behavior')}' in {context}. "
            f"Supported values: {supported}."
        )
