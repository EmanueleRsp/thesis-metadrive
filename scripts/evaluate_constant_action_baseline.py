"""Evaluate a constant-action policy on the frozen ScenarioNet validation panels.

Diagnostic floor for every trained curve: a policy that never learns anything
(holds the brake, coasts, or acts at random) is run through the same panels,
the same evaluation protocol and the same CSV schemas as a training run, so its
`evals.csv` and `eval_episodes.csv` rows are directly comparable with the
intermediate evaluations of any arm that shares the reward configuration.

Run inside the dev container, with the same preset as the arm to compare with:

    uv run --no-sync python scripts/evaluate_constant_action_baseline.py \
        --config-name presets/learnability/sac_a_native \
        +baseline.action=brake \
        analysis.experiment_group=BASELINE-CONSTANT-ACTION-01 \
        env.vectorized.enabled=false experiment.eval_workers=8

`baseline.action` is one of `brake` (steer 0, throttle -1: standstill),
`coast` (steer 0, throttle 0) or `random` (uniform in the action space, seeded).
`+baseline.report_every_s=<seconds>` (default 60) sets how often a progress line
with steps done, episodes done, step rate and an ETA is printed; steps are the
unit because their cost is stable while episode lengths and completion bursts
are not.
The reward configuration of the preset decides which reward the rows report as
`reward`: the native environment reward under `monitor_only`, the scalarized
Rulebook reward under `scalar_reward`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.evaluation_plan import resolve_scenarionet_evaluation_panels
from thesis_rl.runtime.execution.seeding import seed_env_spaces, set_global_seed
from thesis_rl.runtime.final_panels import (
    _append_episode_rows,
    _append_rule_rows,
    _data_abort_fields,
    _metric_fields,
    _panel_fields,
)
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.metadata import save_run_metadata
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_eval_env,
    evaluation_num_workers,
)

CONSTANT_ACTIONS: dict[str, tuple[float, float]] = {
    "brake": (0.0, -1.0),
    "coast": (0.0, 0.0),
}
SUMMARY_KEYS = (
    "route_completion",
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "mean_reward",
    "mean_env_reward",
    "mean_scalar_rule_reward",
)


class ConstantActionPlanner:
    """Planner that ignores the observation and emits a fixed or random action."""

    def __init__(self, action_space: Any, mode: str, seed: int) -> None:
        self.mode = mode
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)
        if mode == "random":
            self._action = None
        elif mode in CONSTANT_ACTIONS:
            self._action = np.asarray(CONSTANT_ACTIONS[mode], dtype=np.float32)
            if self._action.shape != tuple(action_space.shape):
                raise ValueError(
                    f"Constant action shape {self._action.shape} does not match the "
                    f"environment action space {action_space.shape}."
                )
        else:
            raise ValueError(
                f"Unsupported baseline.action {mode!r}; expected one of "
                f"{sorted(CONSTANT_ACTIONS)} or 'random'."
            )

    def predict(self, observation: Any, deterministic: bool = False):
        del observation, deterministic
        if self._action is None:
            low = np.asarray(self.action_space.low, dtype=np.float32)
            high = np.asarray(self.action_space.high, dtype=np.float32)
            return self._rng.uniform(low, high).astype(np.float32), None
        return self._action.copy(), None

    def get_lifecycle(self) -> None:
        return None

    def save(self, checkpoint_path: Any) -> None:
        raise NotImplementedError("The constant-action baseline has nothing to save.")

    def set_env(self, env: Any) -> None:
        del env


class StepProgress:
    """Count environment steps through the evaluator and print a periodic ETA.

    Steps are the right unit for a time estimate: their cost is stable while
    episodes vary from tens to hundreds of steps and, with parallel workers,
    finish in bursts. The wrapper forwards every attribute to the wrapped
    environment; only the stepping methods are intercepted. ``num_envs`` is
    copied as a real attribute because the evaluator reads it without going
    through ``__getattr__``.
    """

    def __init__(self, env: Any, *, label: str, episode_total: int, report_every_s: float) -> None:
        self._env = env
        if hasattr(env, "num_envs"):
            self.num_envs = int(env.num_envs)
        self.label = label
        self.episode_total = int(episode_total)
        self.report_every_s = float(report_every_s)
        self.steps = 0
        self.episodes_done = 0
        self._started = time.monotonic()
        self._last_report = self._started

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any):
        result = self._env.step(action)
        self._count(1)
        return result

    def step_slots(self, actions: Any):
        result = self._env.step_slots(actions)
        self._count(len(actions))
        return result

    def on_episode_progress(self, completed: int, total: int) -> None:
        self.episodes_done = int(completed)
        self.episode_total = int(total)

    def _count(self, n: int) -> None:
        self.steps += int(n)
        now = time.monotonic()
        if now - self._last_report >= self.report_every_s:
            self._last_report = now
            self.report()

    def report(self, *, final: bool = False) -> None:
        elapsed = max(time.monotonic() - self._started, 1e-9)
        rate = self.steps / elapsed
        eta = "n/a"
        if not final and self.episodes_done > 0 and rate > 0:
            # Steps counted so far also belong to the episodes still in flight,
            # roughly one per worker and about half done on average; dividing by
            # completed episodes alone would inflate the mean length and make
            # the estimate grow with the elapsed time.
            in_flight = min(getattr(self, "num_envs", 1), self.episode_total - self.episodes_done)
            mean_len = self.steps / (self.episodes_done + 0.5 * max(in_flight, 0))
            remaining = max(self.episode_total * mean_len - self.steps, 0.0)
            eta = f"{remaining / rate / 60:.1f} min"
        print(
            f"[baseline] {self.label}: steps={self.steps} episodes={self.episodes_done}/"
            f"{self.episode_total} elapsed={elapsed / 60:.1f} min rate={rate:.2f} steps/s "
            f"eta={eta}",
            flush=True,
        )


def _base_csv_fields(cfg: DictConfig, *, mode: str, run_id: str) -> dict[str, Any]:
    return {
        "algorithm": f"constant_action_{mode}",
        "reward_type": str(cfg.reward.type),
        "reward_behavior": str(cfg.reward.behavior),
        "curriculum_name": str(cfg.curriculum.name),
        "rulebook_config": str(cfg.reward.get("rulebook_config", "none")),
        "curriculum_enabled": bool(cfg.curriculum.get("enabled", False)),
        "seed": int(cfg.seed),
        "run_id": run_id,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    baseline_cfg = cfg.get("baseline", {})
    mode = str(baseline_cfg.get("action", "brake")).lower()
    report_every_s = float(baseline_cfg.get("report_every_s", 60.0))
    if bool(cfg.env.get("vectorized", {}).get("enabled", False)):
        raise ValueError("Set env.vectorized.enabled=false: evaluation builds its own workers.")
    set_global_seed(int(cfg.seed))

    run_dir = Path(str(cfg.paths.run_dir))
    run_id = run_dir.name
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path(str(cfg.paths.logs_dir)).mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, artifacts_dir)
    recorder = CSVRecorder(cfg.paths.csv_dir)
    base_fields = _base_csv_fields(cfg, mode=mode, run_id=run_id)

    panels = resolve_scenarionet_evaluation_panels(cfg, final=False)
    if not panels:
        raise RuntimeError("No ScenarioNet validation panels resolved for this configuration.")
    workers = evaluation_num_workers(cfg, final=False)
    batch_id = f"baseline_{mode}"
    summary: dict[str, Any] = {"action": mode, "panels": {}}
    print(f"[baseline] action={mode} panels={[panel.name for panel in panels]} workers={workers}")

    for offset, panel in enumerate(panels, start=1):
        overrides = panel.env_overrides()
        inner_env = build_eval_env(
            cfg, overrides, n_eval_episodes=panel.episode_count, workers=workers
        )
        env = StepProgress(
            inner_env,
            label=f"{mode} {panel.name}",
            episode_total=panel.episode_count,
            report_every_s=report_every_s,
        )
        try:
            seed_env_spaces(env, int(cfg.seed) + 100_000 + offset)
            planner = ConstantActionPlanner(env.action_space, mode, int(cfg.seed) + offset)
            agent = Agent(
                preprocessor=IdentityPreprocessor(),
                planner=planner,
                adapter=IdentityAdapter(**adapter_space_kwargs(env.action_space)),
            )
            metrics = agent.evaluate(
                env=env,
                n_eval_episodes=panel.episode_count,
                deterministic=True,
                base_seed=None,
                return_episode_metrics=True,
                error_priority_base=float(cfg.reward.get("a", 2.01)),
                show_progress=False,
                progress_callback=env.on_episode_progress,
                progress_description=f"Baseline {mode} {panel.name}",
            )
            env.report(final=True)
        finally:
            inner_env.close()
        if not isinstance(metrics, dict) or not metrics.get("per_episode"):
            raise RuntimeError(f"panel {panel.name} produced no complete per-episode metrics")
        common = {
            "eval_id": offset,
            "eval_type": "baseline",
            "scenario_set": panel.scenario_set,
            "chunk_id": 0,
            "stage": "baseline",
            "stage_index": 0,
            "global_step": 0,
            "eval_episodes": panel.episode_count,
            "deterministic": True,
            **_panel_fields(
                panel, batch_id=batch_id, checkpoint_identity=f"constant_action_{mode}"
            ),
            **_data_abort_fields(metrics),
        }
        recorder.append_row(
            "evals.csv",
            {
                **base_fields,
                **common,
                **_metric_fields(metrics),
                "promoted": False,
                "next_stage": "baseline",
            },
        )
        _append_rule_rows(recorder, base_fields, common, metrics)
        _append_episode_rows(
            recorder, base_fields=base_fields, common=common, metrics=metrics, deterministic=True
        )
        panel_summary = {key: metrics.get(key) for key in SUMMARY_KEYS}
        per_episode = metrics["per_episode"]
        lengths = [float(value) for value in per_episode.get("episode_length", [])]
        returns = [float(value) for value in per_episode.get("returns", [])]
        panel_summary["mean_episode_length"] = float(np.mean(lengths)) if lengths else None
        panel_summary["min_return"] = min(returns) if returns else None
        panel_summary["max_return"] = max(returns) if returns else None
        summary["panels"][panel.name] = panel_summary
        print(f"[baseline] {panel.name}: {json.dumps(panel_summary, sort_keys=True)}")

    summary_path = artifacts_dir / "constant_action_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[baseline] summary written to {summary_path}")


if __name__ == "__main__":
    main()
