"""Run the live scalar Rulebook trace on the immutable golden suite."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from thesis_rl.runtime.execution.seeding import set_global_seed
from thesis_rl.runtime.io.eval_artifacts import build_live_final_eval_recorder_factory
from thesis_rl.runtime.io.metadata import save_run_metadata
from thesis_rl.runtime.io.run_logging import configure_logging
from thesis_rl.runtime.wiring.builders import build_env, merge_env_config_with_overrides
from thesis_rl.scenarios.golden import load_golden_scenario_uids


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _assert_finite(value: Any, *, path: str) -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite Rulebook value at {path}: {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, path=f"{path}[{index}]")


def _zero_action(env: Any) -> np.ndarray:
    space = env.action_space
    action = np.zeros(space.shape, dtype=np.float32)
    return np.clip(action, np.asarray(space.low), np.asarray(space.high)).astype(np.float32)


def _build_recorder_factory(
    *, cfg: DictConfig, run_dir: Path, resolved_env_payload: dict[str, Any]
) -> Any:
    """Build the live GIF recorder using the canonical video config paths."""

    return build_live_final_eval_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_config=resolved_env_payload,
        eval_id=1,
        eval_type="final",
        scenario_set="golden_train",
        stage="golden_suite",
        stage_index=0,
        checkpoint_path="none",
        checkpoint_type="diagnostic_zero_policy",
        checkpoint_global_step=0,
    )


@hydra.main(version_base=None, config_path="../../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    manifest_path_raw = cfg.gold_suite.get("manifest_path")
    if manifest_path_raw in (None, "", "null"):
        raise ValueError("gold_suite.manifest_path is required for the golden Rulebook trace.")
    manifest_path = Path(str(manifest_path_raw)).expanduser().resolve()
    expected_uids = load_golden_scenario_uids(manifest_path)
    print(
        f"[gold-rulebook] manifest loaded: {len(expected_uids)} scenarios from {manifest_path}",
        flush=True,
    )
    policy = str(cfg.gold_suite.get("policy", "zero")).strip().lower()
    if policy != "zero":
        raise ValueError("The golden Rulebook trace currently supports only policy=zero.")
    max_steps = int(cfg.gold_suite.get("max_steps", 0))
    if max_steps < 0:
        raise ValueError("gold_suite.max_steps must be non-negative.")

    provider_manifest = cfg.env.provider.get("scenario_uids_file")
    if str(provider_manifest) != str(manifest_path):
        raise ValueError(
            "env.provider.scenario_uids_file must equal gold_suite.manifest_path "
            "to prevent tracing a different scenario set."
        )
    if str(cfg.env.provider.kind).lower() != "fixed_sequence":
        raise ValueError("Golden Rulebook trace requires env.provider.kind=fixed_sequence.")
    if bool(cfg.curriculum.get("enabled", False)):
        raise ValueError("Golden Rulebook trace requires curriculum=disabled.")

    run_dir = Path(str(cfg.paths.run_dir))
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    logs_dir = Path(str(cfg.paths.logs_dir))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, artifacts_dir)
    configure_logging(
        str(cfg.logging.get("level", "INFO")),
        console_level=str(cfg.logging.get("console_level", "WARNING")),
    )
    set_global_seed(int(cfg.seed))
    print("[gold-rulebook] building ScenarioNet environment...", flush=True)

    resolved_env_config = OmegaConf.to_container(
        merge_env_config_with_overrides(cfg.env, {}), resolve=True
    )
    if not isinstance(resolved_env_config, dict):
        raise TypeError("Resolved golden-suite environment config must be a mapping.")
    resolved_env_payload = resolved_env_config.get("config", resolved_env_config)
    if not isinstance(resolved_env_payload, dict):
        raise TypeError("Resolved golden-suite environment payload must be a mapping.")

    env = build_env(cfg)
    print(
        f"[gold-rulebook] environment ready; action_space={env.action_space}",
        flush=True,
    )
    recorder_factory = _build_recorder_factory(
        cfg=cfg,
        run_dir=run_dir,
        resolved_env_payload=resolved_env_payload,
    )

    episodes: list[dict[str, Any]] = []
    termination_counts: Counter[str] = Counter()
    try:
        for episode_index, expected_uid in enumerate(expected_uids):
            print(
                f"[gold-rulebook] episode {episode_index + 1}/{len(expected_uids)} "
                f"expected_uid={expected_uid}",
                flush=True,
            )
            observation, reset_info = env.reset()
            observed_uid = str(reset_info.get("scenario_uid", ""))
            if observed_uid != expected_uid:
                raise RuntimeError(
                    "Golden-suite order mismatch at episode "
                    f"{episode_index + 1}: expected {expected_uid!r}, got {observed_uid!r}."
                )
            recorder = recorder_factory(
                {
                    "episode_idx": episode_index,
                    "episode_id": episode_index + 1,
                    "scenario_seed": None,
                    "scenario_uid": observed_uid,
                    "deterministic": True,
                }
            )
            step_count = 0
            scalar_return = 0.0
            margin_values: list[float] = []
            episode_collision = False
            episode_out_of_road = False
            done = False
            truncated = False
            last_info: dict[str, Any] = {}
            while not (done or truncated):
                if max_steps > 0 and step_count >= max_steps:
                    raise RuntimeError(
                        f"Golden scenario {observed_uid!r} exceeded max_steps={max_steps}."
                    )
                action = _zero_action(env)
                next_observation, reward, done, truncated, step_info = env.step(action)
                if not isinstance(step_info, dict):
                    raise TypeError("Golden Rulebook trace requires mapping step info.")
                if "rulebook" not in step_info or "rule_reward_vector" not in step_info:
                    raise RuntimeError(
                        f"Rulebook payload missing for {observed_uid!r} step {step_count}."
                    )
                if "scalar_rule_reward" not in step_info:
                    raise RuntimeError(
                        f"Scalar Rulebook reward missing for {observed_uid!r} step {step_count}."
                    )
                _assert_finite(step_info["rulebook"], path="rulebook")
                _assert_finite(step_info["rule_reward_vector"], path="rule_reward_vector")
                _assert_finite(step_info["scalar_rule_reward"], path="scalar_rule_reward")
                _assert_finite(float(reward), path="reward")
                margin_values.extend(float(value) for value in step_info["rule_reward_vector"])
                scalar_return += float(reward)
                episode_collision = episode_collision or bool(
                    step_info.get("collision") or step_info.get("crash_vehicle")
                )
                episode_out_of_road = episode_out_of_road or bool(step_info.get("out_of_road"))
                recorder.record_step(
                    env=env,
                    step_index=step_count,
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    truncated=bool(truncated),
                    step_info=step_info,
                )
                observation = next_observation
                last_info = dict(step_info)
                step_count += 1
                if step_count % 100 == 0:
                    print(
                        f"[gold-rulebook] {observed_uid} step={step_count}",
                        flush=True,
                    )
            reason = str(
                last_info.get("termination_reason")
                or ("truncated" if truncated else "terminated")
            )
            termination_counts[reason] += 1
            episode_metrics = {
                "scenario_uid": observed_uid,
                "source": reset_info.get("source"),
                "arm": reset_info.get("arm") or reset_info.get("scenario_arm"),
                "steps": step_count,
                "return": scalar_return,
                "terminated": bool(done),
                "truncated": bool(truncated),
                "termination_reason": reason,
                "collision": episode_collision,
                "out_of_road": episode_out_of_road,
                "min_rule_margin": min(margin_values) if margin_values else None,
                "max_rule_margin": max(margin_values) if margin_values else None,
            }
            episode_metrics["artifacts"] = recorder.finalize_episode(
                episode_metrics=episode_metrics
            )
            episodes.append(episode_metrics)
            print(
                f"[gold-rulebook] completed {episode_index + 1}/{len(expected_uids)} "
                f"uid={observed_uid} steps={step_count} reason={reason} "
                f"collision={episode_collision} out_of_road={episode_out_of_road}",
                flush=True,
            )
    finally:
        env.close()

    report = {
        "schema": "golden_rulebook_live_trace_v1",
        "manifest_path": str(manifest_path),
        "manifest_scenario_count": len(expected_uids),
        "observed_scenario_count": len(episodes),
        "policy": policy,
        "rulebook_version": str(cfg.rulebook.version),
        "scalarization_mode": str(cfg.scalarization.mode),
        "termination_counts": dict(sorted(termination_counts.items())),
        "episodes": _json_safe(episodes),
    }
    report_path = artifacts_dir / "golden_rulebook_trace.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[gold-rulebook] report written: {report_path}", flush=True)
    print(json.dumps({"report": str(report_path), "episodes": len(episodes)}, sort_keys=True))


if __name__ == "__main__":
    main()
