from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from thesis_rl.preprocessors.base import BasePreprocessor
from thesis_rl.agents.base import BasePlanner
from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.agents.types import Transition


class _LiveEventLogHandler(logging.Handler):
    """Capture log records into the Live monitor event deque."""

    def __init__(self, sink: deque[str]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            module_name = record.name.split(".")[-1]
            message = record.getMessage()
            self._sink.appendleft(f"[{record.levelname}] {module_name}: {message}")
        except Exception:
            self.handleError(record)


class Agent:
    """Composable agent that applies preprocessor -> planner -> adapter."""

    def __init__(self, preprocessor: BasePreprocessor, planner: BasePlanner, adapter: BaseAdapter) -> None:
        self.preprocessor = preprocessor
        self.planner = planner
        self.adapter = adapter

    def train(
        self,
        env: Any,
        chunk_timesteps: int,
        global_total_timesteps: int,
        global_steps_done: int,
        deterministic: bool = False,
        log_interval: int = 1000,
        reset_seed: int | None = None,
    ) -> dict[str, float | int]:
        '''Train the agent in the given environment for a specified number of timesteps.
        Args:
            env: The environment to train in. Must have `reset()` and `step()` methods.
            chunk_timesteps: Total number of environment steps to train for.
            global_total_timesteps: Total number of timesteps for the entire training run.
            global_steps_done: Number of timesteps already completed in the entire training run.
            deterministic: Whether to use deterministic actions during training.
            log_interval: Interval (in timesteps) at which to log training metrics.
            reset_seed: Optional seed for environment reset at the start of training.
        Returns:
            Dictionary with chunk-level summary metrics (episodes, moving averages, update stats, fps).
        '''

        ###################
        ###### SETUP ######
        ###################

        # Initialize planner lifecycle and adapter for training
        lifecycle = self.planner.get_lifecycle()
        lifecycle.begin_training(
            chunk_timesteps=chunk_timesteps,
            global_total_timesteps=global_total_timesteps,
            global_steps_done=global_steps_done,
        )
        self.adapter.begin_training()

        # Reset environment and preprocessor state at the start of training
        self.preprocessor.reset()
        if reset_seed is not None:
            obs, _ = env.reset(seed=int(reset_seed))
        else:
            obs, _ = env.reset()

        # Initialize timers (for logging)
        start_time = time.time()
        # Initialize episode tracking variables
        episodes = 0
        episode_len = 0
        episode_scalar_reward = 0.0
        episode_success = False
        episode_collision = False
        episode_out_of_road = False
        episode_route_completion = 0.0
        # Use deques to track recent episode lengths and rewards for moving average metrics
        recent_episode_lens: deque[int] = deque(maxlen=100)
        recent_episode_rewards: deque[float] = deque(maxlen=100)
        chunk_episode_success: list[float] = []
        chunk_episode_collision: list[float] = []
        chunk_episode_out_of_road: list[float] = []
        chunk_episode_route_completion: list[float] = []
        event_logs: deque[str] = deque(maxlen=8)
        monitor_log_handler = _LiveEventLogHandler(event_logs)
        monitor_logger_names = (
            "thesis_rl.envs.wrappers",
            "thesis_rl.rulebook.evaluator",
        )
        monitor_logger_restore_state: list[tuple[logging.Logger, int, bool]] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            expand=True,
        )
        progress_task = progress.add_task("Training chunk", total=chunk_timesteps)

        def _build_monitor_table(
            current_step: int,
            total_step: int,
            chunk_step: int,
            elapsed_seconds: float,
            total_episodes: int,
            mean_episode_len: float,
            mean_episode_reward: float,
            latest_actor_loss: float,
            latest_critic_loss: float,
            latest_learning_rate: float,
            total_updates: int,
        ) -> Table:
            table = Table(title="Training Monitor", expand=True)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            table.add_row("Chunk steps", f"{chunk_step}/{chunk_timesteps}")
            table.add_row("Total steps", f"{current_step}/{total_step}")
            table.add_row("Episodes", str(total_episodes))
            table.add_row("FPS", str(int(chunk_step / max(elapsed_seconds, 1e-9))))
            table.add_row("Elapsed (s)", str(int(elapsed_seconds)))
            table.add_row("ep_len_mean", f"{mean_episode_len:.2f}")
            table.add_row("ep_rew_mean", f"{mean_episode_reward:.2f}")
            table.add_row("actor_loss", f"{latest_actor_loss:.3g}")
            table.add_row("critic_loss", f"{latest_critic_loss:.3g}")
            table.add_row("learning_rate", f"{latest_learning_rate:.3g}")
            table.add_row("n_updates", str(total_updates))
            return table

        def _build_logs_panel(logs: deque[str]) -> Panel:
            if not logs:
                content = "No events yet"
            else:
                content = "\n".join(logs)
            return Panel(content, title="Events", expand=True)

        ######################
        ###### TRAINING ######
        ######################

        # Main training loop over environment steps
        for logger_name in monitor_logger_names:
            logger = logging.getLogger(logger_name)
            monitor_logger_restore_state.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.addHandler(monitor_log_handler)

        try:
            with Live(refresh_per_second=8, screen=False) as live:
                for step in range(1, chunk_timesteps + 1):

                    ###### STEP ######

                    # Preprocess observation
                    processed_obs = self.preprocessor(obs)
                    # Get action from planner (TO.ANALYZE)
                    planner_output = lifecycle.act(processed_obs, deterministic=deterministic)
                    # Transform planner output to environment action space
                    action = self.adapter(planner_output)

                    # Step environment with action and observe transition
                    next_obs, scalar_reward, done, truncated, step_info = env.step(action)

                    ###### UPDATE ######

                    # Update episode tracking variables
                    terminated = bool(done or truncated)
                    episode_len += 1
                    episode_scalar_reward += float(scalar_reward)
                    episode_success = episode_success or self._extract_success(step_info)
                    episode_collision = episode_collision or self._extract_collision(step_info)
                    episode_out_of_road = episode_out_of_road or self._extract_out_of_road(step_info)
                    episode_route_completion = max(
                        episode_route_completion,
                        self._extract_route_completion(step_info),
                    )

                    # Record transition
                    next_processed_obs = self.preprocessor(next_obs)
                    if not isinstance(step_info, dict):
                        raise TypeError(
                            f"Expected `step_info` to be dict, got {type(step_info).__name__}."
                        )
                    lifecycle.observe_transition(
                        Transition(
                            observation=processed_obs,
                            env_action=np.asarray(action, dtype=np.float32),
                            buffer_action=lifecycle.to_buffer_action(np.asarray(action, dtype=np.float32)),
                            scalar_reward=float(scalar_reward),
                            terminated=bool(done),
                            truncated=bool(truncated),
                            next_observation=next_processed_obs,
                            terminal_observation=next_processed_obs if terminated else None,
                            info=dict(step_info),
                        )
                    )

                    # (Maybe) Update planner and adapter
                    lifecycle.maybe_update()
                    self.adapter.maybe_update()

                    # If episode ended
                    if terminated:
                        lifecycle.on_episode_end()
                        event_logs.appendleft(
                            f"Episode {episodes + 1} ended | len={episode_len} reward={episode_scalar_reward:.2f}"
                        )
                        # Increment episode count and record episode metrics
                        episodes += 1
                        recent_episode_lens.append(episode_len)
                        recent_episode_rewards.append(episode_scalar_reward)
                        chunk_episode_success.append(1.0 if episode_success else 0.0)
                        chunk_episode_collision.append(1.0 if episode_collision else 0.0)
                        chunk_episode_out_of_road.append(1.0 if episode_out_of_road else 0.0)
                        chunk_episode_route_completion.append(float(episode_route_completion))

                        # Reset episode tracking variables
                        episode_len = 0
                        episode_scalar_reward = 0.0
                        episode_success = False
                        episode_collision = False
                        episode_out_of_road = False
                        episode_route_completion = 0.0

                        # Reset environment and preprocessor state for next episode
                        self.preprocessor.reset()
                        obs, _ = env.reset()
                    else:
                        obs = next_obs

                    if self._extract_collision(step_info):
                        event_logs.appendleft("Collision detected")
                    if self._extract_out_of_road(step_info):
                        event_logs.appendleft("Out of road detected")

                    ###### LOGGING ######
                    progress.update(progress_task, completed=step)

                    elapsed = max(time.time() - start_time, 1e-9)
                    ep_len_mean = float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
                    ep_rew_mean = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0
                    actor_loss = float(getattr(lifecycle, "last_actor_loss", float("nan")))
                    critic_loss = float(getattr(lifecycle, "last_critic_loss", float("nan")))
                    learning_rate = float(getattr(lifecycle, "last_learning_rate", float("nan")))
                    n_updates = int(getattr(lifecycle, "update_count", 0))

                    should_render = log_interval <= 0 or step % log_interval == 0 or step == chunk_timesteps
                    if should_render:
                        monitor = _build_monitor_table(
                            current_step=global_steps_done + step,
                            total_step=global_total_timesteps,
                            chunk_step=step,
                            elapsed_seconds=elapsed,
                            total_episodes=episodes,
                            mean_episode_len=ep_len_mean,
                            mean_episode_reward=ep_rew_mean,
                            latest_actor_loss=actor_loss,
                            latest_critic_loss=critic_loss,
                            latest_learning_rate=learning_rate,
                            total_updates=n_updates,
                        )
                        layout = Group(progress, monitor, _build_logs_panel(event_logs))
                        live.update(layout)
        finally:
            for logger, original_level, original_propagate in monitor_logger_restore_state:
                logger.removeHandler(monitor_log_handler)
                logger.setLevel(original_level)
                logger.propagate = original_propagate

        # Finalize training lifecycle
        lifecycle.end_training()
        self.adapter.end_training()
        elapsed = max(time.time() - start_time, 1e-9)
        ep_success_rate = (
            float(np.mean(chunk_episode_success))
            if chunk_episode_success
            else None
        )
        ep_collision_rate = (
            float(np.mean(chunk_episode_collision))
            if chunk_episode_collision
            else None
        )
        ep_out_of_road_rate = (
            float(np.mean(chunk_episode_out_of_road))
            if chunk_episode_out_of_road
            else None
        )
        ep_route_completion_mean = (
            float(np.mean(chunk_episode_route_completion))
            if chunk_episode_route_completion
            else None
        )
        return {
            "episodes": int(episodes),
            "ep_len_mean": float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0,
            "ep_rew_mean": float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0,
            "ep_success_rate": ep_success_rate,
            "ep_collision_rate": ep_collision_rate,
            "ep_out_of_road_rate": ep_out_of_road_rate,
            "ep_route_completion_mean": ep_route_completion_mean,
            "actor_loss": float(getattr(lifecycle, "last_actor_loss", float("nan"))),
            "critic_loss": float(getattr(lifecycle, "last_critic_loss", float("nan"))),
            "learning_rate": float(getattr(lifecycle, "last_learning_rate", float("nan"))),
            "n_updates": int(getattr(lifecycle, "update_count", 0)),
            "fps": float(chunk_timesteps / elapsed),
            "elapsed_seconds": float(elapsed),
        }

    def predict(self, observation: Any, deterministic: bool = False):
        processed_obs = self.preprocessor(observation)
        planner_output, state = self.planner.predict(processed_obs, deterministic=deterministic)
        action = self.adapter(planner_output)
        return action, state

    def evaluate(
        self,
        env: Any,
        n_eval_episodes: int,
        deterministic: bool = False,
        base_seed: int | None = None,
        return_episode_metrics: bool = False,
        error_priority_base: float = 2.01,
    ) -> dict[str, Any]:
        '''Evaluate the agent in the given environment for a specified number of episodes.
        Args:
            env: The environment to evaluate in. Must have `reset()` and `step()` methods.
            n_eval_episodes: Number of episodes to evaluate for.
            deterministic: Whether to use deterministic actions during evaluation.
        Returns:
            Aggregate metrics for the full evaluation set. If `return_episode_metrics=True`,
            the output also includes a `per_episode` section with raw episode vectors.
        '''
        # Validate `n_eval_episodes`
        if int(n_eval_episodes) <= 0:
            raise ValueError("`n_eval_episodes` must be > 0.")

        # Initialize episode-level metric trackers
        episode_returns: list[float] = []
        episode_saturation_max: list[float] = []
        episode_collision: list[float] = []
        episode_out_of_road: list[float] = []
        episode_success: list[float] = []
        episode_route_completion: list[float] = []
        episode_top_rule_violation_rate: list[float] = []
        episode_lengths: list[int] = []
        episode_timeout: list[float] = []
        episode_error_values: list[float] = []
        episode_violation_patterns: list[str] = []
        episode_violated_rule_names: list[str] = []
        episode_env_returns: list[float] = []
        episode_scalar_rule_returns: list[float | None] = []
        episode_rule_rewards_by_rule: list[dict[str, float]] = []
        all_rule_names: set[str] = set()
        rule_priority_by_name: dict[str, int] = {}
        per_rule_episode_min_margins: dict[str, list[float]] = {}
        per_rule_violation_count: dict[str, int] = {}

        # Loop over evaluation episodes
        for episode_idx in range(n_eval_episodes):
            # Reset preprocessor and environment state at the start of each episode
            self.preprocessor.reset()
            if base_seed is not None:
                obs, _ = env.reset(seed=int(base_seed) + episode_idx)
            else:
                obs, _ = env.reset()

            # Initialize episode tracking variables
            done = False
            truncated = False
            ep_return = 0.0
            ep_sat_max = 0.0
            ep_collision = False
            ep_out_of_road = False
            ep_success = False
            ep_route_completion = 0.0
            ep_step_count = 0
            ep_top_rule_violating_steps = 0
            ep_rule_min_margin: dict[str, float] = {}
            ep_rule_reward_sum_by_rule: dict[str, float] = {}
            ep_env_return = 0.0
            ep_scalar_rule_return = 0.0
            ep_has_scalar_rule_reward = False

            # Loop until episode ends
            while not (done or truncated):
                # Get action
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, scalar_reward, done, truncated, step_info = env.step(action)
                ep_return += float(scalar_reward)
                ep_step_count += 1
                if isinstance(step_info, dict):
                    env_reward = step_info.get("env_reward")
                    if isinstance(env_reward, (int, float, np.floating)):
                        ep_env_return += float(env_reward)
                    else:
                        ep_env_return += float(scalar_reward)
                    scalar_rule_reward = step_info.get("scalar_rule_reward")
                    if isinstance(scalar_rule_reward, (int, float, np.floating)):
                        ep_scalar_rule_return += float(scalar_rule_reward)
                        ep_has_scalar_rule_reward = True
                else:
                    ep_env_return += float(scalar_reward)

                # Extract rule violation and saturation info from `step_info` for episode-level metrics
                sat_summary = self._extract_saturation_summary(step_info)
                if sat_summary is not None:
                    _sat_rule, sat_ratio = sat_summary
                    ep_sat_max = max(ep_sat_max, sat_ratio)
                ep_collision = ep_collision or self._extract_collision(step_info)
                ep_out_of_road = ep_out_of_road or self._extract_out_of_road(step_info)
                ep_success = ep_success or self._extract_success(step_info)
                ep_route_completion = max(ep_route_completion, self._extract_route_completion(step_info))
                if self._has_top_rule_violation(step_info):
                    ep_top_rule_violating_steps += 1
                for rule_name, rule_priority, margin in self._extract_rule_margins(step_info):
                    all_rule_names.add(rule_name)
                    if rule_name not in rule_priority_by_name:
                        rule_priority_by_name[rule_name] = int(rule_priority)
                    ep_rule_reward_sum_by_rule[rule_name] = float(
                        ep_rule_reward_sum_by_rule.get(rule_name, 0.0) + float(margin)
                    )
                    ep_rule_min_margin[rule_name] = min(
                        float(ep_rule_min_margin.get(rule_name, float("inf"))),
                        float(margin),
                    )

            # Record episode metrics
            episode_returns.append(ep_return)
            episode_saturation_max.append(ep_sat_max)
            episode_collision.append(1.0 if ep_collision else 0.0)
            episode_out_of_road.append(1.0 if ep_out_of_road else 0.0)
            episode_success.append(1.0 if ep_success else 0.0)
            episode_route_completion.append(ep_route_completion)
            episode_lengths.append(ep_step_count)
            episode_timeout.append(1.0 if bool(truncated) else 0.0)
            episode_env_returns.append(float(ep_env_return))
            episode_scalar_rule_returns.append(
                float(ep_scalar_rule_return) if ep_has_scalar_rule_reward else None
            )
            episode_rule_rewards_by_rule.append(dict(sorted(ep_rule_reward_sum_by_rule.items())))
            if ep_step_count > 0:
                episode_top_rule_violation_rate.append(ep_top_rule_violating_steps / ep_step_count)
            else:
                episode_top_rule_violation_rate.append(0.0)

            violated_in_episode: list[tuple[str, int]] = []
            for rule_name, min_margin in ep_rule_min_margin.items():
                per_rule_episode_min_margins.setdefault(rule_name, []).append(float(min_margin))
                if float(min_margin) < 0.0:
                    per_rule_violation_count[rule_name] = int(per_rule_violation_count.get(rule_name, 0) + 1)
                    violated_in_episode.append((rule_name, int(rule_priority_by_name.get(rule_name, 0))))

            if violated_in_episode:
                violated_in_episode.sort(key=lambda item: (item[1], item[0]))
                violation_pattern = "+".join(name for name, _prio in violated_in_episode)
                episode_violated_rule_names.append(violation_pattern)
            else:
                violation_pattern = "none"
                episode_violated_rule_names.append("none")
            episode_violation_patterns.append(violation_pattern)

            if ep_rule_min_margin:
                p_max = max(int(rule_priority_by_name.get(name, 0)) for name in ep_rule_min_margin)
                ev_episode = 0.0
                for rule_name, min_margin in ep_rule_min_margin.items():
                    priority = int(rule_priority_by_name.get(rule_name, 0))
                    weight = float(error_priority_base) ** float(p_max - priority)
                    ev_episode += weight * max(0.0, -float(min_margin))
                episode_error_values.append(float(ev_episode))
            else:
                episode_error_values.append(0.0)

        counterexample_rate = (
            float(np.mean([1.0 if pattern != "none" else 0.0 for pattern in episode_violation_patterns]))
            if episode_violation_patterns
            else 0.0
        )
        violated_rules_ratio = (
            float(sum(1 for rule in all_rule_names if int(per_rule_violation_count.get(rule, 0)) > 0))
            / float(max(len(all_rule_names), 1))
        )
        unique_violation_patterns = int(len(set(episode_violation_patterns)))

        per_rule_rows: list[dict[str, Any]] = []
        for rule_name in sorted(all_rule_names, key=lambda name: (rule_priority_by_name.get(name, 0), name)):
            margins = per_rule_episode_min_margins.get(rule_name, [])
            if margins:
                mean_margin = float(np.mean(margins))
                min_margin = float(np.min(margins))
                max_margin = float(np.max(margins))
            else:
                mean_margin = 0.0
                min_margin = 0.0
                max_margin = 0.0
            violation_count = int(per_rule_violation_count.get(rule_name, 0))
            per_rule_rows.append(
                {
                    "rule_name": rule_name,
                    "rule_priority": int(rule_priority_by_name.get(rule_name, 0)),
                    "violated": bool(violation_count > 0),
                    "violation_rate": float(violation_count / max(len(episode_returns), 1)),
                    "violation_count": violation_count,
                    "mean_margin": mean_margin,
                    "min_margin": min_margin,
                    "max_margin": max_margin,
                }
            )

        scalar_rule_available_values = [value for value in episode_scalar_rule_returns if value is not None]

        metrics: dict[str, Any] = {
            "mean_reward": float(np.mean(episode_returns)),
            "std_reward": float(np.std(episode_returns)),
            "mean_env_reward": float(np.mean(episode_env_returns)) if episode_env_returns else 0.0,
            "std_env_reward": float(np.std(episode_env_returns)) if episode_env_returns else 0.0,
            "mean_scalar_rule_reward": (
                float(np.mean(scalar_rule_available_values))
                if scalar_rule_available_values
                else None
            ),
            "std_scalar_rule_reward": (
                float(np.std(scalar_rule_available_values))
                if scalar_rule_available_values
                else None
            ),
            "mean_rule_saturation_max": float(np.mean(episode_saturation_max)),
            "collision_rate": float(np.mean(episode_collision)),
            "collision_rate_std": float(np.std(episode_collision)),
            "out_of_road_rate": float(np.mean(episode_out_of_road)),
            "success_rate": float(np.mean(episode_success)),
            "success_rate_std": float(np.std(episode_success)),
            "route_completion": float(np.mean(episode_route_completion)),
            "top_rule_violation_rate": float(np.mean(episode_top_rule_violation_rate)),
            "avg_error_value": float(np.mean(episode_error_values)) if episode_error_values else 0.0,
            "max_error_value": float(np.max(episode_error_values)) if episode_error_values else 0.0,
            "counterexample_rate": counterexample_rate,
            "violated_rules_ratio": float(violated_rules_ratio),
            "unique_violation_patterns": unique_violation_patterns,
            "per_rule": per_rule_rows,
        }
        if return_episode_metrics:
            metrics["per_episode"] = {
                "returns": episode_returns,
                "env_returns": episode_env_returns,
                "scalar_rule_returns": episode_scalar_rule_returns,
                "rule_rewards_by_rule": episode_rule_rewards_by_rule,
                "saturation_max": episode_saturation_max,
                "collision": episode_collision,
                "out_of_road": episode_out_of_road,
                "success": episode_success,
                "route_completion": episode_route_completion,
                "top_rule_violation_rate": episode_top_rule_violation_rate,
                "episode_length": episode_lengths,
                "timeout": episode_timeout,
                "error_value": episode_error_values,
                "violation_pattern": episode_violation_patterns,
                "violated_rules": episode_violated_rule_names,
            }
        return metrics

    @staticmethod
    def _extract_rule_margins(step_info: Any) -> list[tuple[str, int, float]]:
        if not isinstance(step_info, dict):
            return []
        names = step_info.get("rule_metadata", {}).get("rule_names") if isinstance(step_info.get("rule_metadata"), dict) else None
        priorities = step_info.get("rule_metadata", {}).get("priorities") if isinstance(step_info.get("rule_metadata"), dict) else None
        margins = step_info.get("rule_reward_vector")
        if not isinstance(names, list) or not isinstance(priorities, list):
            return []
        if not isinstance(margins, (list, tuple, np.ndarray)):
            return []
        size = min(len(names), len(priorities), len(margins))
        rows: list[tuple[str, int, float]] = []
        for idx in range(size):
            rows.append((str(names[idx]), int(priorities[idx]), float(margins[idx])))
        return rows

    @staticmethod
    def _extract_saturation_summary(step_info: Any) -> tuple[str, float] | None:
        if not isinstance(step_info, dict):
            return None

        rule_metadata = step_info.get("rule_metadata")
        if not isinstance(rule_metadata, dict):
            return None

        ratios = rule_metadata.get("saturation_ratio_by_rule")
        if not isinstance(ratios, dict) or not ratios:
            return None

        best_rule, best_ratio = max(
            ((str(name), float(value)) for name, value in ratios.items()),
            key=lambda item: item[1],
        )
        return best_rule, best_ratio

    @staticmethod
    def _extract_collision(step_info: Any) -> bool:
        if not isinstance(step_info, dict):
            return False
        keys = (
            "crash",
            "crash_vehicle",
            "crash_object",
            "crash_building",
            "crash_human",
            "collision",
        )
        return any(bool(step_info.get(key, False)) for key in keys)

    @staticmethod
    def _extract_out_of_road(step_info: Any) -> bool:
        if not isinstance(step_info, dict):
            return False
        return bool(step_info.get("out_of_road", False))

    @staticmethod
    def _extract_success(step_info: Any) -> bool:
        if not isinstance(step_info, dict):
            return False
        return bool(step_info.get("arrive_dest", False) or step_info.get("success", False))

    @staticmethod
    def _extract_route_completion(step_info: Any) -> float:
        if not isinstance(step_info, dict):
            return 0.0

        for key in ("route_completion", "route_completion_ratio", "progress"):
            value = step_info.get(key)
            if value is not None:
                return float(value)

        if bool(step_info.get("arrive_dest", False) or step_info.get("success", False)):
            return 1.0
        return 0.0

    @staticmethod
    def _has_top_rule_violation(step_info: Any) -> bool:
        if not isinstance(step_info, dict):
            return False

        violation_vec = step_info.get("rule_violation_vector")
        if isinstance(violation_vec, (list, tuple, np.ndarray)) and len(violation_vec) > 0:
            return float(violation_vec[0]) > 0.0

        rule_reward_vec = step_info.get("rule_reward_vector")
        if isinstance(rule_reward_vec, (list, tuple, np.ndarray)) and len(rule_reward_vec) > 0:
            # Margins are positive when satisfied and negative when violated.
            return float(rule_reward_vec[0]) < 0.0

        return False

    @staticmethod
    def adapter_checkpoint_path(checkpoint_path: str | Path) -> Path:
        """Derive adapter checkpoint path from planner checkpoint stem.

        Examples:
            - checkpoints/baseline_td3 -> checkpoints/baseline_td3.adapter.pt
            - checkpoints/baseline_td3.zip -> checkpoints/baseline_td3.adapter.pt
        """
        checkpoint = Path(checkpoint_path)
        planner_stem = checkpoint.with_suffix("") if checkpoint.suffix == ".zip" else checkpoint
        return planner_stem.parent / f"{planner_stem.name}.adapter.pt"

    def save(self, checkpoint_path: str | Path) -> None:
        '''Save planner and adapter state to the specified checkpoint path.'''
        # Save planner state
        self.planner.save(checkpoint_path)
        # Save adapter state if the adapter is trainable
        if bool(getattr(self.adapter, "requires_training", False)):
            adapter_ckpt = self.adapter_checkpoint_path(checkpoint_path)
            self.adapter.save(str(adapter_ckpt))

    def load_adapter(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        """Load adapter state derived from planner checkpoint path.

        If the adapter is not trainable, this is a no-op.
        """
        if not bool(getattr(self.adapter, "requires_training", False)):
            return

        adapter_ckpt = self.adapter_checkpoint_path(checkpoint_path)
        if not adapter_ckpt.exists():
            if strict:
                raise FileNotFoundError(
                    f"Adapter checkpoint not found for trainable adapter: {adapter_ckpt}"
                )
            return
        self.adapter.load(str(adapter_ckpt))
