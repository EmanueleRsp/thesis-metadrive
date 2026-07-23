from __future__ import annotations

import logging
import time
import math
from collections import deque
from collections.abc import Callable, Mapping
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

from thesis_rl.agent.planners.core.utils import count_envs
from thesis_rl.agent.preprocessors.interfaces.base import BasePreprocessor
from thesis_rl.agent.planners.interfaces.planner import BasePlanner
from thesis_rl.agent.adapters.interfaces.base import BaseAdapter
from thesis_rl.agent.types import Transition
from thesis_rl.agent.transition_boundary import normalize_vector_transition_boundary
from thesis_rl.contracts.checkpoint_manifest import CheckpointManifest
from thesis_rl.contracts.reward_semantics import write_reward_semantics_sidecar
from thesis_rl.sb3_extensions.checkpointing import (
    CheckpointGeneration,
    publish_checkpoint_generation,
)


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


def _format_episode_event(
    *,
    episode_number: int,
    env_index: int,
    episode_length: int,
    reward: float,
    reason: str,
    route_completion: float,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Format the stable completed-episode line used by every train runner."""

    context_suffix = ""
    if context:
        rendered_context = " ".join(
            f"{key}={value}" for key, value in context.items() if value is not None
        )
        if rendered_context:
            context_suffix = f" | {rendered_context}"
    return (
        f"Episode {int(episode_number)} env={int(env_index)} ended | "
        f"len={int(episode_length)} reward={float(reward):.2f} | reason={reason} "
        f"route_completion={float(route_completion):.2f}{context_suffix}"
    )


class _ParallelEvaluationEpisode:
    """Slot-local evaluation state used by the deterministic vector runner."""

    _METADATA_KEYS = (
        "scenario_uid",
        "scenario_id",
        "source",
        "split",
        "arm",
        "scenario_arm",
        "worker_id",
        "sampling_mode",
        "requested_arm",
        "source_cell_fallback",
    )

    def __init__(
        self,
        *,
        episode_idx: int,
        scenario_seed: int | None,
        reset_info: Any,
        artifact_recorder: Any | None,
    ) -> None:
        self.episode_idx = int(episode_idx)
        self.scenario_seed = scenario_seed
        self.artifact_recorder = artifact_recorder
        self.metadata = {
            key: reset_info.get(key)
            for key in self._METADATA_KEYS
            if isinstance(reset_info, dict) and key in reset_info
        }
        self.last_step_info: Any = None
        self.ep_return = 0.0
        self.ep_sat_max = 0.0
        self.ep_collision = False
        self.ep_out_of_road = False
        self.ep_success = False
        self.ep_route_completion = 0.0
        self.ep_step_count = 0
        self.ep_top_rule_violating_steps = 0
        self.ep_rule_min_margin: dict[str, float] = {}
        self.ep_rule_reward_sum_by_rule: dict[str, float] = {}
        self.ep_env_return = 0.0
        self.ep_scalar_rule_return = 0.0
        self.ep_has_scalar_rule_reward = False
        self.ep_hybrid_return = 0.0
        self.ep_has_hybrid_reward = False
        self.rule_priority_by_name: dict[str, int] = {}

    def observe_step(
        self,
        agent: "Agent",
        *,
        env: Any,
        observation: Any,
        next_observation: Any,
        action: Any,
        scalar_reward: Any,
        done: bool,
        truncated: bool,
        step_info: Any,
    ) -> None:
        self.last_step_info = step_info
        self.ep_return += float(scalar_reward)
        self.ep_step_count += 1
        if isinstance(step_info, dict):
            env_reward = step_info.get("env_reward")
            if isinstance(env_reward, (int, float, np.floating)):
                self.ep_env_return += float(env_reward)
            else:
                self.ep_env_return += float(scalar_reward)
            scalar_rule_reward = step_info.get("scalar_rule_reward")
            if isinstance(scalar_rule_reward, (int, float, np.floating)):
                self.ep_scalar_rule_return += float(scalar_rule_reward)
                self.ep_has_scalar_rule_reward = True
            hybrid_reward = step_info.get("hybrid_reward")
            if isinstance(hybrid_reward, (int, float, np.floating)):
                self.ep_hybrid_return += float(hybrid_reward)
                self.ep_has_hybrid_reward = True
        else:
            self.ep_env_return += float(scalar_reward)

        sat_summary = agent._extract_saturation_summary(step_info)
        if sat_summary is not None:
            _sat_rule, sat_ratio = sat_summary
            self.ep_sat_max = max(self.ep_sat_max, sat_ratio)
        self.ep_collision = self.ep_collision or agent._extract_collision(step_info)
        self.ep_out_of_road = self.ep_out_of_road or agent._extract_out_of_road(step_info)
        self.ep_success = self.ep_success or agent._extract_success(step_info)
        self.ep_route_completion = max(
            self.ep_route_completion, agent._extract_route_completion(step_info)
        )
        if agent._has_top_rule_violation(step_info):
            self.ep_top_rule_violating_steps += 1
        for rule_name, rule_priority, margin in agent._extract_rule_margins(step_info):
            self.rule_priority_by_name[rule_name] = int(rule_priority)
            self.ep_rule_reward_sum_by_rule[rule_name] = float(
                self.ep_rule_reward_sum_by_rule.get(rule_name, 0.0) + float(margin)
            )
            self.ep_rule_min_margin[rule_name] = min(
                float(self.ep_rule_min_margin.get(rule_name, float("inf"))),
                float(margin),
            )
        if self.artifact_recorder is not None:
            self.artifact_recorder.record_step(
                env=env,
                step_index=self.ep_step_count - 1,
                observation=observation,
                next_observation=next_observation,
                action=np.asarray(action, dtype=np.float32),
                reward=float(scalar_reward),
                done=bool(done),
                truncated=bool(truncated),
                step_info=step_info,
            )

    def finalize(
        self,
        agent: "Agent",
        *,
        done: bool,
        truncated: bool,
        error_priority_base: float,
    ) -> dict[str, Any]:
        violated_in_episode: list[tuple[str, int]] = []
        for rule_name, min_margin in self.ep_rule_min_margin.items():
            if float(min_margin) < 0.0:
                violated_in_episode.append(
                    (rule_name, int(self.rule_priority_by_name.get(rule_name, 0)))
                )
        violated_in_episode.sort(key=lambda item: (item[1], item[0]))
        violation_pattern = (
            "+".join(name for name, _priority in violated_in_episode)
            if violated_in_episode
            else "none"
        )
        if self.ep_rule_min_margin:
            p_max = max(
                int(self.rule_priority_by_name.get(name, 0)) for name in self.ep_rule_min_margin
            )
            error_value = sum(
                float(error_priority_base)
                ** float(p_max - self.rule_priority_by_name.get(rule_name, 0))
                * max(0.0, -float(min_margin))
                for rule_name, min_margin in self.ep_rule_min_margin.items()
            )
        else:
            error_value = 0.0

        episode_top_rule_rate = (
            self.ep_top_rule_violating_steps / self.ep_step_count if self.ep_step_count > 0 else 0.0
        )
        episode_metrics = {
            "reward": float(self.ep_return),
            "env_reward": float(self.ep_env_return),
            "scalar_rule_reward": (
                float(self.ep_scalar_rule_return) if self.ep_has_scalar_rule_reward else None
            ),
            "hybrid_reward": float(self.ep_hybrid_return) if self.ep_has_hybrid_reward else None,
            "episode_length": int(self.ep_step_count),
            "success": bool(self.ep_success),
            "collision": bool(self.ep_collision),
            "out_of_road": bool(self.ep_out_of_road),
            "timeout": bool(truncated),
            "route_completion": float(self.ep_route_completion),
            "top_rule_violation_rate": float(episode_top_rule_rate),
            "error_value": float(error_value),
            "violated_rules": violation_pattern,
            "violation_pattern": violation_pattern,
        }
        artifact_payload = (
            self.artifact_recorder.finalize_episode(episode_metrics=episode_metrics)
            if self.artifact_recorder is not None
            else {}
        )
        if isinstance(self.last_step_info, dict):
            self.metadata.update(
                {
                    key: self.last_step_info[key]
                    for key in self._METADATA_KEYS
                    if key in self.last_step_info
                }
            )
            self.metadata["termination_reason"] = self.last_step_info.get("termination_reason")
        self.metadata["terminated"] = bool(done)
        self.metadata["truncated"] = bool(truncated)
        return {
            "episode_idx": self.episode_idx,
            "reward": float(self.ep_return),
            "env_reward": float(self.ep_env_return),
            "scalar_rule_reward": (
                float(self.ep_scalar_rule_return) if self.ep_has_scalar_rule_reward else None
            ),
            "hybrid_reward": float(self.ep_hybrid_return) if self.ep_has_hybrid_reward else None,
            "episode_length": int(self.ep_step_count),
            "success": 1.0 if self.ep_success else 0.0,
            "collision": 1.0 if self.ep_collision else 0.0,
            "out_of_road": 1.0 if self.ep_out_of_road else 0.0,
            "route_completion": float(self.ep_route_completion),
            "top_rule_violation_rate": float(episode_top_rule_rate),
            "saturation_max": float(self.ep_sat_max),
            "timeout": 1.0 if truncated else 0.0,
            "error_value": float(error_value),
            "violation_pattern": violation_pattern,
            "violated_rules": violation_pattern,
            "rule_rewards_by_rule": dict(sorted(self.ep_rule_reward_sum_by_rule.items())),
            "rule_min_margins": dict(self.ep_rule_min_margin),
            "rule_priorities": dict(self.rule_priority_by_name),
            "video_path": artifact_payload.get("video_path"),
            "video_authoritative_path": artifact_payload.get("video_authoritative_path"),
            "video_manifest_path": artifact_payload.get("video_manifest_path"),
            "trajectory_log_path": artifact_payload.get("trajectory_log_path"),
            "video_recorded_live": bool(artifact_payload.get("video_recorded_live", False)),
            "replay_warning": artifact_payload.get("replay_warning"),
            "scenario_metadata": dict(self.metadata),
        }


class Agent:
    """Composable agent that applies preprocessor -> planner -> adapter."""

    def __init__(
        self,
        preprocessor: BasePreprocessor,
        planner: BasePlanner,
        adapter: BaseAdapter,
        ema_alpha: float | None = None,
        checkpoint_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.planner = planner
        self.adapter = adapter
        # EMA alpha for loss smoothing; default 0.1 if not provided
        self.ema_alpha = float(ema_alpha) if ema_alpha is not None else 0.1
        self.checkpoint_identity = (
            None if checkpoint_identity is None else dict(checkpoint_identity)
        )

    def set_checkpoint_identity(self, identity: Mapping[str, Any] | None) -> None:
        """Set the immutable reward identity written beside runtime checkpoints."""

        self.checkpoint_identity = None if identity is None else dict(identity)

    def train(
        self,
        env: Any,
        chunk_timesteps: int,
        global_total_timesteps: int,
        global_steps_done: int,
        stage_name: str | None = None,
        deterministic: bool = False,
        log_interval: int = 1000,
        reset_seed: int | None = None,
        reset_seed_fn: Callable[[int], int | None] | None = None,
        episode_end_callback: Callable[[Any, int, dict[str, Any]], None] | None = None,
        before_episode_reset_callback: Callable[[Any, int], None] | None = None,
        episode_context_callback: Callable[[Any, dict[str, Any]], dict[str, Any] | None]
        | None = None,
        monitor_extra_rows_callback: Callable[[], list[tuple[str, str]]] | None = None,
        monitor_event_poll_callback: Callable[[], list[str]] | None = None,
        live_extra_renderables_callback: Callable[[], list[Any]] | None = None,
    ) -> dict[str, Any]:
        """Train the agent in the given environment for a specified number of timesteps.
        Args:
            env: The environment to train in. Must have `reset()` and `step()` methods.
            chunk_timesteps: Total number of environment steps to train for.
            global_total_timesteps: Total number of timesteps for the entire training run.
            global_steps_done: Number of timesteps already completed in the entire training run.
            deterministic: Whether to use deterministic actions during training.
            log_interval: Interval (in timesteps) at which to log training metrics.
            reset_seed: Optional seed for environment reset at the start of training.
            reset_seed_fn: Optional callable returning the environment seed for each
                episode reset, where index 0 is the first reset of this chunk.
            episode_end_callback: Optional callback invoked after each completed
                episode as ``callback(env, episode_index, metrics)``.
            before_episode_reset_callback: Optional callback invoked immediately
                before every reset, including the first reset (index zero).
            episode_context_callback: Optional callback providing metadata to
                append to each completed-episode monitor event.
            monitor_extra_rows_callback: Optional callback providing live
                monitor rows as ``(metric, value)`` pairs.
            monitor_event_poll_callback: Optional callback returning background
                event-log messages to merge into the training Live view.
            live_extra_renderables_callback: Optional callback returning Rich
                renderables displayed below the training monitor.
        Returns:
            Dictionary with chunk-level summary metrics (episodes, moving averages, update stats, fps).
        """

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
        reset_seeds_used: list[int] = []
        if before_episode_reset_callback is not None:
            before_episode_reset_callback(env, 0)
        first_reset_seed = reset_seed_fn(0) if reset_seed_fn is not None else reset_seed
        if first_reset_seed is not None:
            reset_seeds_used.append(int(first_reset_seed))
            obs, _ = env.reset(seed=int(first_reset_seed))
        else:
            obs, _ = env.reset()

        # Initialize timers (for logging)
        start_time = time.time()
        # Initialize episode tracking variables
        episodes = 0
        episode_len = 0
        episode_scalar_reward = 0.0
        episode_env_reward = 0.0
        episode_scalar_rule_reward = 0.0
        episode_hybrid_reward = 0.0
        episode_has_scalar_rule_reward = False
        episode_has_hybrid_reward = False
        episode_success = False
        episode_collision = False
        episode_out_of_road = False
        episode_route_completion = 0.0
        # Use deques to track recent episode lengths and rewards for moving average metrics
        recent_episode_lens: deque[int] = deque(maxlen=100)
        recent_episode_rewards: deque[float] = deque(maxlen=100)
        recent_episode_env_rewards: deque[float] = deque(maxlen=100)
        recent_episode_scalar_rule_rewards: deque[float] = deque(maxlen=100)
        recent_episode_hybrid_rewards: deque[float] = deque(maxlen=100)
        chunk_episode_success: list[float] = []
        chunk_episode_collision: list[float] = []
        chunk_episode_out_of_road: list[float] = []
        chunk_episode_route_completion: list[float] = []
        event_logs: deque[str] = deque(maxlen=8)
        monitor_log_handler = _LiveEventLogHandler(event_logs)
        # Avoid log-stream interference with Rich Live in multiprocessing/PTY contexts.
        # Keep live metrics table, but do not attach runtime loggers to the event panel.
        monitor_logger_names: tuple[str, ...] = ()
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

        # EMA smoothing for losses (to display stable trends in monitor)
        ema_alpha = float(getattr(self, "ema_alpha", 0.1))
        ema_actor_loss: float = float("nan")
        ema_critic_loss: float = float("nan")
        monitor_title = "Training Monitor"
        if stage_name:
            monitor_title = f"Training Monitor ({stage_name})"

        def _build_monitor_table(
            current_step: int,
            total_step: int,
            chunk_step: int,
            elapsed_seconds: float,
            total_episodes: int,
            mean_episode_len: float,
            std_episode_len: float,
            mean_episode_reward: float,
            std_episode_reward: float,
            latest_actor_loss: float,
            latest_critic_loss: float,
            total_update_calls: int,
            total_gradient_steps: int,
            latest_actor_loss_ema: float,
            latest_critic_loss_ema: float,
        ) -> Table:
            table = Table(title=monitor_title, expand=True)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            table.add_row("Chunk env steps", f"{chunk_step}/{chunk_timesteps}")
            table.add_row("Run env steps", f"{current_step}/{total_step}")
            table.add_row("Episodes", str(total_episodes))
            table.add_row("FPS", str(int(chunk_step / max(elapsed_seconds, 1e-9))))
            table.add_row("Elapsed (s)", str(int(elapsed_seconds)))
            table.add_row("Length (per episode)", f"{mean_episode_len:.2f} ± {std_episode_len:.2f}")
            table.add_row(
                "Reward (per episode)", f"{mean_episode_reward:.2f} ± {std_episode_reward:.2f}"
            )
            table.add_row("Actor loss (EMA)", f"{latest_actor_loss_ema:.3g}")
            table.add_row("Critic loss (EMA)", f"{latest_critic_loss_ema:.3g}")
            table.add_row("Update calls (chunk)", str(total_update_calls))
            table.add_row("Gradient steps (chunk)", str(total_gradient_steps))
            if monitor_extra_rows_callback is not None:
                for metric, value in monitor_extra_rows_callback():
                    table.add_row(str(metric), str(value))
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
            with Live(
                refresh_per_second=8,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
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
                    if isinstance(step_info, dict):
                        env_reward = step_info.get("env_reward")
                        episode_env_reward += (
                            float(env_reward)
                            if isinstance(env_reward, (int, float, np.floating))
                            else float(scalar_reward)
                        )
                        scalar_rule_reward = step_info.get("scalar_rule_reward")
                        if isinstance(scalar_rule_reward, (int, float, np.floating)):
                            episode_scalar_rule_reward += float(scalar_rule_reward)
                            episode_has_scalar_rule_reward = True
                        hybrid_reward = step_info.get("hybrid_reward")
                        if isinstance(hybrid_reward, (int, float, np.floating)):
                            episode_hybrid_reward += float(hybrid_reward)
                            episode_has_hybrid_reward = True
                    else:
                        episode_env_reward += float(scalar_reward)
                    episode_success = episode_success or self._extract_success(step_info)
                    episode_collision = episode_collision or self._extract_collision(step_info)
                    episode_out_of_road = episode_out_of_road or self._extract_out_of_road(
                        step_info
                    )
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
                            buffer_action=lifecycle.to_buffer_action(
                                np.asarray(action, dtype=np.float32)
                            ),
                            scalar_reward=float(scalar_reward),
                            terminated=bool(done),
                            truncated=bool(truncated),
                            next_observation=next_processed_obs,
                            terminal_observation=next_processed_obs
                            if terminated or truncated
                            else None,
                            info=dict(step_info),
                        )
                    )

                    # (Maybe) Update planner and adapter
                    lifecycle.maybe_update()
                    self.adapter.maybe_update()

                    # If episode ended
                    if terminated:
                        lifecycle.on_episode_end()
                        # Determine termination reason (priority: success > collision > out_of_road > timeout)
                        if episode_success:
                            reason = "success"
                        elif episode_collision:
                            reason = "collision"
                        elif episode_out_of_road:
                            reason = "out_of_road"
                        else:
                            reason = "timeout"
                        # Increment episode count and record episode metrics
                        episodes += 1
                        recent_episode_lens.append(episode_len)
                        recent_episode_rewards.append(episode_scalar_reward)
                        recent_episode_env_rewards.append(episode_env_reward)
                        if episode_has_scalar_rule_reward:
                            recent_episode_scalar_rule_rewards.append(episode_scalar_rule_reward)
                        if episode_has_hybrid_reward:
                            recent_episode_hybrid_rewards.append(episode_hybrid_reward)
                        chunk_episode_success.append(1.0 if episode_success else 0.0)
                        chunk_episode_collision.append(1.0 if episode_collision else 0.0)
                        chunk_episode_out_of_road.append(1.0 if episode_out_of_road else 0.0)
                        chunk_episode_route_completion.append(float(episode_route_completion))

                        episode_critic_loss = float(
                            getattr(lifecycle, "last_critic_loss", float("nan"))
                        )
                        episode_metrics: dict[str, Any] = {
                            "length": episode_len,
                            "reward": episode_scalar_reward,
                            "success": episode_success,
                            "collision": episode_collision,
                            "out_of_road": episode_out_of_road,
                            "route_completion": episode_route_completion,
                            "update_calls": int(getattr(lifecycle, "update_count", 0)),
                        }
                        if math.isfinite(episode_critic_loss):
                            episode_metrics["critic_loss"] = episode_critic_loss
                        episode_learning_potential = float(
                            getattr(lifecycle, "last_learning_potential", float("nan"))
                        )
                        if math.isfinite(episode_learning_potential):
                            episode_metrics["learning_potential"] = episode_learning_potential
                        if episode_end_callback is not None:
                            episode_end_callback(
                                env,
                                episodes,
                                episode_metrics,
                            )
                        episode_context: Mapping[str, Any] | None = None
                        if episode_context_callback is not None:
                            episode_context = episode_context_callback(env, step_info)
                        event_logs.appendleft(
                            _format_episode_event(
                                episode_number=episodes,
                                env_index=0,
                                episode_length=episode_len,
                                reward=episode_scalar_reward,
                                reason=reason,
                                route_completion=episode_route_completion,
                                context=episode_context,
                            )
                        )

                        # Reset episode tracking variables
                        episode_len = 0
                        episode_scalar_reward = 0.0
                        episode_env_reward = 0.0
                        episode_scalar_rule_reward = 0.0
                        episode_hybrid_reward = 0.0
                        episode_has_scalar_rule_reward = False
                        episode_has_hybrid_reward = False
                        episode_success = False
                        episode_collision = False
                        episode_out_of_road = False
                        episode_route_completion = 0.0

                        # Reset environment and preprocessor state for next episode
                        self.preprocessor.reset()
                        if before_episode_reset_callback is not None:
                            before_episode_reset_callback(env, episodes)
                        next_reset_seed = (
                            reset_seed_fn(episodes) if reset_seed_fn is not None else None
                        )
                        if next_reset_seed is not None:
                            reset_seeds_used.append(int(next_reset_seed))
                            obs, _ = env.reset(seed=int(next_reset_seed))
                        else:
                            obs, _ = env.reset()
                    else:
                        obs = next_obs

                    ###### LOGGING ######
                    progress.update(progress_task, completed=step)

                    elapsed = max(time.time() - start_time, 1e-9)
                    ep_len_mean = (
                        float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
                    )
                    ep_len_std = float(np.std(recent_episode_lens)) if recent_episode_lens else 0.0
                    ep_rew_mean = (
                        float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0
                    )
                    ep_rew_std = (
                        float(np.std(recent_episode_rewards)) if recent_episode_rewards else 0.0
                    )
                    actor_loss = float(getattr(lifecycle, "last_actor_loss", float("nan")))
                    critic_loss = float(getattr(lifecycle, "last_critic_loss", float("nan")))
                    update_calls = int(getattr(lifecycle, "update_count", 0))
                    gradient_steps_total = int(getattr(lifecycle, "gradient_step_count", 0))

                    # Update EMA for losses
                    raw_actor = actor_loss
                    raw_critic = critic_loss
                    if not math.isnan(raw_actor):
                        ema_actor_loss = (
                            raw_actor
                            if math.isnan(ema_actor_loss)
                            else (ema_alpha * raw_actor + (1 - ema_alpha) * ema_actor_loss)
                        )
                    if not math.isnan(raw_critic):
                        ema_critic_loss = (
                            raw_critic
                            if math.isnan(ema_critic_loss)
                            else (ema_alpha * raw_critic + (1 - ema_alpha) * ema_critic_loss)
                        )

                    should_render = (
                        log_interval <= 0 or step % log_interval == 0 or step == chunk_timesteps
                    )
                    extra_events = (
                        monitor_event_poll_callback() if monitor_event_poll_callback else []
                    )
                    event_logs.extendleft(reversed(extra_events))
                    if should_render:
                        monitor = _build_monitor_table(
                            current_step=global_steps_done + step,
                            total_step=global_total_timesteps,
                            chunk_step=step,
                            elapsed_seconds=elapsed,
                            total_episodes=episodes,
                            mean_episode_len=ep_len_mean,
                            std_episode_len=ep_len_std,
                            mean_episode_reward=ep_rew_mean,
                            std_episode_reward=ep_rew_std,
                            latest_actor_loss=actor_loss,
                            latest_critic_loss=critic_loss,
                            total_update_calls=update_calls,
                            total_gradient_steps=gradient_steps_total,
                            latest_actor_loss_ema=ema_actor_loss,
                            latest_critic_loss_ema=ema_critic_loss,
                        )
                        extra_renderables = (
                            live_extra_renderables_callback()
                            if live_extra_renderables_callback
                            else []
                        )
                        layout = Group(
                            progress,
                            monitor,
                            *extra_renderables,
                            _build_logs_panel(event_logs),
                        )
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
        ep_success_rate = float(np.mean(chunk_episode_success)) if chunk_episode_success else None
        ep_collision_rate = (
            float(np.mean(chunk_episode_collision)) if chunk_episode_collision else None
        )
        ep_out_of_road_rate = (
            float(np.mean(chunk_episode_out_of_road)) if chunk_episode_out_of_road else None
        )
        ep_route_completion_mean = (
            float(np.mean(chunk_episode_route_completion))
            if chunk_episode_route_completion
            else None
        )
        # Compute std and 95% CI for episode rewards and lengths
        ep_len_final_mean = float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
        ep_len_final_std = float(np.std(recent_episode_lens)) if recent_episode_lens else 0.0
        ep_len_ci_95 = (
            1.96 * ep_len_final_std / np.sqrt(len(recent_episode_lens))
            if recent_episode_lens and len(recent_episode_lens) > 1
            else 0.0
        )

        ep_rew_final_mean = (
            float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0
        )
        ep_rew_final_std = float(np.std(recent_episode_rewards)) if recent_episode_rewards else 0.0
        ep_rew_ci_95 = (
            1.96 * ep_rew_final_std / np.sqrt(len(recent_episode_rewards))
            if recent_episode_rewards and len(recent_episode_rewards) > 1
            else 0.0
        )
        ep_env_rew_final_mean = (
            float(np.mean(recent_episode_env_rewards)) if recent_episode_env_rewards else None
        )
        ep_scalar_rule_rew_final_mean = (
            float(np.mean(recent_episode_scalar_rule_rewards))
            if recent_episode_scalar_rule_rewards
            else None
        )
        ep_hybrid_rew_final_mean = (
            float(np.mean(recent_episode_hybrid_rewards)) if recent_episode_hybrid_rewards else None
        )

        return {
            "episodes": int(episodes),
            "ep_len_mean": ep_len_final_mean,
            "ep_len_std": ep_len_final_std,
            "ep_len_ci_95": float(ep_len_ci_95),
            "ep_rew_mean": ep_rew_final_mean,
            "ep_rew_std": ep_rew_final_std,
            "ep_rew_ci_95": float(ep_rew_ci_95),
            "ep_env_rew_mean": ep_env_rew_final_mean,
            "ep_scalar_rule_rew_mean": ep_scalar_rule_rew_final_mean,
            "ep_hybrid_rew_mean": ep_hybrid_rew_final_mean,
            "ep_success_rate": ep_success_rate,
            "ep_collision_rate": ep_collision_rate,
            "ep_out_of_road_rate": ep_out_of_road_rate,
            "ep_route_completion_mean": ep_route_completion_mean,
            "actor_loss": float(getattr(lifecycle, "last_actor_loss", float("nan"))),
            "critic_loss": float(getattr(lifecycle, "last_critic_loss", float("nan"))),
            "learning_potential": (
                float(np.mean(lifecycle.learning_potential_values))
                if getattr(lifecycle, "learning_potential_values", [])
                else None
            ),
            "actor_loss_ema": float(ema_actor_loss) if not math.isnan(ema_actor_loss) else None,
            "critic_loss_ema": float(ema_critic_loss) if not math.isnan(ema_critic_loss) else None,
            "learning_rate": float(getattr(lifecycle, "last_learning_rate", float("nan"))),
            "update_calls": int(getattr(lifecycle, "update_count", 0)),
            "n_updates": int(getattr(lifecycle, "gradient_step_count", 0)),
            "fps": float(chunk_timesteps / elapsed),
            "elapsed_seconds": float(elapsed),
            "chunk_steps_actual": int(chunk_timesteps),
            "train_reset_seed_first": reset_seeds_used[0] if reset_seeds_used else None,
            "train_reset_seed_last": reset_seeds_used[-1] if reset_seeds_used else None,
            "train_reset_seed_unique_count": len(set(reset_seeds_used)),
        }

    def train_vectorized(
        self,
        env: Any,
        chunk_timesteps: int,
        global_total_timesteps: int,
        global_steps_done: int,
        stage_name: str | None = None,
        deterministic: bool = False,
        log_interval: int = 1000,
        reset_seed_fn: Callable[[int], int | None] | None = None,
        vector_episode_end_callback: Callable[
            [Any, list[int], list[dict[str, Any]]], Mapping[int, Any]
        ]
        | None = None,
        initial_observations: Any | None = None,
        monitor_extra_rows_callback: Callable[[], list[tuple[str, str]]] | None = None,
        monitor_event_poll_callback: Callable[[], list[str]] | None = None,
        live_extra_renderables_callback: Callable[[], list[Any]] | None = None,
        slow_step_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train with a vectorized env, counting total collected transitions.

        ACL vector environments expose terminal observations without auto-reset.
        Their parent callback receives completed slot metrics after the learner
        transition is stored and must commit the ACL outcome, configure the next
        selections, reset only those slots, and return ``slot -> observation``.
        Generic vector environments retain their existing worker auto-reset.
        """
        n_envs = count_envs(env)
        if n_envs <= 1:
            return self.train(
                env=env,
                chunk_timesteps=chunk_timesteps,
                global_total_timesteps=global_total_timesteps,
                global_steps_done=global_steps_done,
                stage_name=stage_name,
                deterministic=deterministic,
                log_interval=log_interval,
                reset_seed_fn=reset_seed_fn,
                monitor_extra_rows_callback=monitor_extra_rows_callback,
                monitor_event_poll_callback=monitor_event_poll_callback,
                live_extra_renderables_callback=live_extra_renderables_callback,
            )
        if bool(getattr(self.adapter, "requires_training", False)):
            raise ValueError("Vectorized training currently supports only stateless adapters.")

        lifecycle = self.planner.get_lifecycle()
        lifecycle.begin_training(
            chunk_timesteps=chunk_timesteps,
            global_total_timesteps=global_total_timesteps,
            global_steps_done=global_steps_done,
        )
        self.adapter.begin_training()
        self.preprocessor.reset()

        reset_seeds_used: list[int] = []
        if reset_seed_fn is not None:
            first_seed = reset_seed_fn(0)
            if first_seed is not None:
                # For SubprocVecEnv + MetaDrive, each worker has its own scenario window
                # [start_index, start_index + num_scenarios). Seed each worker inside its
                # own window to keep deterministic behavior and avoid out-of-range asserts.
                seeded = False
                try:
                    if hasattr(env, "env_method"):
                        bounds = env.env_method("get_worker_seed_bounds")
                        start_indices = [int(item[0]) for item in bounds]
                        scenario_counts = [int(item[1]) for item in bounds]
                    else:
                        start_indices = [int(v) for v in env.get_attr("start_index")]
                        scenario_counts = [int(v) for v in env.get_attr("num_scenarios")]
                    worker_seeds: list[int] = []
                    for rank, (start_index, count) in enumerate(
                        zip(start_indices, scenario_counts)
                    ):
                        if count <= 0:
                            raise ValueError(f"Worker {rank} has invalid num_scenarios={count}.")
                        offset = (int(first_seed) + rank - start_index) % count
                        worker_seeds.append(int(start_index + offset))
                    if len(worker_seeds) == n_envs and hasattr(env, "_seeds"):
                        env._seeds = worker_seeds  # type: ignore[attr-defined]
                        reset_seeds_used.extend(worker_seeds)
                        seeded = True
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to assign per-worker reset seeds for vectorized env: %s", exc
                    )
                if not seeded:
                    # Keep metadata visibility even if worker seeding is unavailable.
                    reset_seeds_used.append(int(first_seed))
        try:
            obs = env.reset() if initial_observations is None else initial_observations
        except EOFError as exc:
            raise RuntimeError(
                "A SubprocVecEnv worker crashed during reset(). "
                "With MetaDrive, this is commonly caused by multiprocessing start-method incompatibility. "
                "Use env.vectorized.start_method=spawn (forkserver may crash workers)."
            ) from exc

        start_time = time.time()
        episodes = 0
        collected_steps = 0
        episode_len = np.zeros(n_envs, dtype=np.int64)
        episode_scalar_reward = np.zeros(n_envs, dtype=np.float64)
        episode_env_reward = np.zeros(n_envs, dtype=np.float64)
        episode_scalar_rule_reward = np.zeros(n_envs, dtype=np.float64)
        episode_hybrid_reward = np.zeros(n_envs, dtype=np.float64)
        episode_has_scalar_rule_reward = np.zeros(n_envs, dtype=bool)
        episode_has_hybrid_reward = np.zeros(n_envs, dtype=bool)
        episode_success = np.zeros(n_envs, dtype=bool)
        episode_collision = np.zeros(n_envs, dtype=bool)
        episode_out_of_road = np.zeros(n_envs, dtype=bool)
        episode_route_completion = np.zeros(n_envs, dtype=np.float64)

        recent_episode_lens: deque[int] = deque(maxlen=100)
        recent_episode_rewards: deque[float] = deque(maxlen=100)
        recent_episode_env_rewards: deque[float] = deque(maxlen=100)
        recent_episode_scalar_rule_rewards: deque[float] = deque(maxlen=100)
        recent_episode_hybrid_rewards: deque[float] = deque(maxlen=100)
        chunk_episode_success: list[float] = []
        chunk_episode_collision: list[float] = []
        chunk_episode_out_of_road: list[float] = []
        chunk_episode_route_completion: list[float] = []
        event_logs: deque[str] = deque(maxlen=8)
        monitor_log_handler = _LiveEventLogHandler(event_logs)
        # Avoid log-stream interference with Rich Live in multiprocessing/PTY contexts.
        # Keep live metrics table, but do not attach runtime loggers to the event panel.
        monitor_logger_names: tuple[str, ...] = ()
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
        ema_alpha = float(getattr(self, "ema_alpha", 0.1))
        ema_actor_loss: float = float("nan")
        ema_critic_loss: float = float("nan")
        phase_seconds = {
            "observation": 0.0,
            "encoder_action": 0.0,
            "env_step": 0.0,
            "transition_collection": 0.0,
            "learner_update": 0.0,
            "logging_callback": 0.0,
            "acl_reset_callback": 0.0,
            "worker_reset": 0.0,
            "worker_wrapped_env_step": 0.0,
            "worker_video_info_enrichment": 0.0,
            "rulebook_snapshot": 0.0,
            "rulebook_evaluator": 0.0,
            "rulebook_scalarization": 0.0,
            "rulebook_observation_refresh": 0.0,
            "rulebook_info_and_diagnostics": 0.0,
        }

        def _preprocess_batch(batch: np.ndarray) -> np.ndarray:
            return np.stack([self.preprocessor(item) for item in batch]).astype(np.float32)

        def _adapt_batch(batch: np.ndarray) -> np.ndarray:
            return np.stack([self.adapter(item) for item in batch]).astype(np.float32)

        monitor_title = "Training Monitor"
        if stage_name:
            monitor_title = f"Training Monitor ({stage_name})"

        def _table() -> Table:
            elapsed = max(time.time() - start_time, 1e-9)
            chunk_env_steps = min(collected_steps, chunk_timesteps)
            run_env_steps = global_steps_done + collected_steps
            table = Table(title=monitor_title, expand=True)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            table.add_row("Chunk env steps", f"{chunk_env_steps}/{chunk_timesteps}")
            table.add_row("Run env steps", f"{run_env_steps}/{global_total_timesteps}")
            table.add_row("Episodes", str(episodes))
            table.add_row("FPS", str(int(collected_steps / elapsed)))
            table.add_row("Elapsed (s)", str(int(elapsed)))
            ep_len_mean = float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
            ep_len_std = float(np.std(recent_episode_lens)) if recent_episode_lens else 0.0
            ep_rew_mean = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0
            ep_rew_std = float(np.std(recent_episode_rewards)) if recent_episode_rewards else 0.0
            table.add_row("Length (per episode)", f"{ep_len_mean:.2f} ± {ep_len_std:.2f}")
            table.add_row("Reward (per episode)", f"{ep_rew_mean:.2f} ± {ep_rew_std:.2f}")
            table.add_row("Actor loss (EMA)", f"{ema_actor_loss:.3g}")
            table.add_row("Critic loss (EMA)", f"{ema_critic_loss:.3g}")
            table.add_row("Update calls (chunk)", str(int(getattr(lifecycle, "update_count", 0))))
            table.add_row(
                "Gradient steps (chunk)", str(int(getattr(lifecycle, "gradient_step_count", 0)))
            )
            if monitor_extra_rows_callback is not None:
                for metric, value in monitor_extra_rows_callback():
                    table.add_row(str(metric), str(value))
            return table

        def _logs_panel() -> Panel:
            content = "\n".join(event_logs) if event_logs else "No events yet"
            return Panel(content, title="Events", expand=True)

        for logger_name in monitor_logger_names:
            logger = logging.getLogger(logger_name)
            monitor_logger_restore_state.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.addHandler(monitor_log_handler)

        try:
            with Live(
                refresh_per_second=8,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                while collected_steps < chunk_timesteps:
                    iteration_started = time.perf_counter()
                    iteration_accounted_seconds = 0.0
                    phase_started = time.perf_counter()
                    processed_obs = _preprocess_batch(np.asarray(obs))
                    phase_elapsed = time.perf_counter() - phase_started
                    phase_seconds["observation"] += phase_elapsed
                    iteration_accounted_seconds += phase_elapsed
                    phase_started = time.perf_counter()
                    planner_output, buffer_actions = lifecycle.act_batch(
                        processed_obs,
                        deterministic=deterministic,
                    )
                    actions = _adapt_batch(planner_output)
                    phase_elapsed = time.perf_counter() - phase_started
                    phase_seconds["encoder_action"] += phase_elapsed
                    iteration_accounted_seconds += phase_elapsed
                    phase_started = time.perf_counter()
                    next_obs, rewards, dones, infos = env.step(actions)
                    phase_elapsed = time.perf_counter() - phase_started
                    phase_seconds["env_step"] += phase_elapsed
                    iteration_accounted_seconds += phase_elapsed
                    phase_started = time.perf_counter()
                    infos = [dict(info) for info in infos]
                    for worker_id, info in enumerate(infos):
                        worker_timing = info.get("_thesis_worker_timing_seconds", {})
                        if isinstance(worker_timing, Mapping):
                            phase_seconds["worker_wrapped_env_step"] += float(
                                worker_timing.get("wrapped_env_step", 0.0)
                            )
                            phase_seconds["worker_video_info_enrichment"] += float(
                                worker_timing.get("video_info_enrichment", 0.0)
                            )
                        rulebook_timing = info.get("_thesis_rulebook_timing_seconds", {})
                        if isinstance(rulebook_timing, Mapping):
                            for name, value in rulebook_timing.items():
                                phase_seconds.setdefault(f"rulebook_{name}", 0.0)
                                phase_seconds[f"rulebook_{name}"] += float(value)
                            if slow_step_callback is not None:
                                slow_step_callback(
                                    global_steps_done + collected_steps,
                                    {
                                        "worker_id": int(worker_id),
                                        "info": info,
                                    },
                                )
                    rewards = np.asarray(rewards, dtype=np.float32)
                    dones = np.asarray(dones, dtype=bool)
                    for info in infos:
                        terminal_observation = info.get("final_observation")
                        if terminal_observation is None:
                            terminal_observation = info.get("terminal_observation")
                        if terminal_observation is not None:
                            processed_final_observation = self.preprocessor(terminal_observation)
                            info["final_observation"] = processed_final_observation
                            info["terminal_observation"] = processed_final_observation

                    terminated, truncated, final_observations = (
                        normalize_vector_transition_boundary(
                            dones=dones,
                            infos=infos,
                            next_observations=np.asarray(next_obs),
                            terminated=np.asarray(
                                [
                                    bool(
                                        info.get(
                                            "terminated",
                                            done and not info.get("TimeLimit.truncated", False),
                                        )
                                    )
                                    for info, done in zip(infos, dones, strict=True)
                                ],
                                dtype=bool,
                            ),
                            truncated=np.asarray(
                                [
                                    bool(
                                        info.get(
                                            "truncated",
                                            done and info.get("TimeLimit.truncated", False),
                                        )
                                    )
                                    for info, done in zip(infos, dones, strict=True)
                                ],
                                dtype=bool,
                            ),
                        )
                    )

                    episode_len += 1
                    episode_scalar_reward += rewards.astype(np.float64)
                    for idx, info in enumerate(infos):
                        env_reward = info.get("env_reward")
                        episode_env_reward[idx] += (
                            float(env_reward)
                            if isinstance(env_reward, (int, float, np.floating))
                            else float(rewards[idx])
                        )
                        scalar_rule_reward = info.get("scalar_rule_reward")
                        if isinstance(scalar_rule_reward, (int, float, np.floating)):
                            episode_scalar_rule_reward[idx] += float(scalar_rule_reward)
                            episode_has_scalar_rule_reward[idx] = True
                        hybrid_reward = info.get("hybrid_reward")
                        if isinstance(hybrid_reward, (int, float, np.floating)):
                            episode_hybrid_reward[idx] += float(hybrid_reward)
                            episode_has_hybrid_reward[idx] = True
                        episode_success[idx] = episode_success[idx] or self._extract_success(info)
                        episode_collision[idx] = episode_collision[idx] or self._extract_collision(
                            info
                        )
                        episode_out_of_road[idx] = episode_out_of_road[
                            idx
                        ] or self._extract_out_of_road(info)
                        episode_route_completion[idx] = max(
                            float(episode_route_completion[idx]),
                            self._extract_route_completion(info),
                        )

                    next_processed_obs = _preprocess_batch(np.asarray(next_obs))
                    for idx, info in enumerate(infos):
                        if dones[idx]:
                            info["final_observation"] = self.preprocessor(final_observations[idx])
                            info["terminal_observation"] = info["final_observation"]
                            next_processed_obs[idx] = info["final_observation"]
                    lifecycle.observe_transition_batch(
                        observations=processed_obs,
                        buffer_actions=buffer_actions,
                        rewards=rewards,
                        dones=dones,
                        next_observations=next_processed_obs,
                        infos=infos,
                        terminated=terminated,
                        truncated=truncated,
                    )
                    phase_elapsed = time.perf_counter() - phase_started
                    phase_seconds["transition_collection"] += phase_elapsed
                    iteration_accounted_seconds += phase_elapsed
                    phase_started = time.perf_counter()
                    lifecycle.maybe_update()
                    self.adapter.maybe_update()
                    phase_elapsed = time.perf_counter() - phase_started
                    phase_seconds["learner_update"] += phase_elapsed
                    iteration_accounted_seconds += phase_elapsed

                    collected_steps += n_envs
                    done_indices = np.flatnonzero(dones)
                    acl_episode_payloads = [
                        {
                            "worker_id": int(idx),
                            "episode_id": infos[idx].get("acl_episode_id"),
                            "episode_length": int(episode_len[idx]),
                            "episode_return": float(episode_scalar_reward[idx]),
                            "terminated": bool(terminated[idx]),
                            "truncated": bool(truncated[idx]),
                            "metrics": {
                                "reward": float(episode_scalar_reward[idx]),
                                "success": bool(episode_success[idx]),
                                "collision": bool(episode_collision[idx]),
                                "out_of_road": bool(episode_out_of_road[idx]),
                                "route_completion": float(episode_route_completion[idx]),
                                "env_reward": float(episode_env_reward[idx]),
                                "scalar_rule_reward": (
                                    float(episode_scalar_rule_reward[idx])
                                    if episode_has_scalar_rule_reward[idx]
                                    else None
                                ),
                                "hybrid_reward": (
                                    float(episode_hybrid_reward[idx])
                                    if episode_has_hybrid_reward[idx]
                                    else None
                                ),
                                "termination_reason": (
                                    "success"
                                    if episode_success[idx]
                                    else "collision"
                                    if episode_collision[idx]
                                    else "out_of_road"
                                    if episode_out_of_road[idx]
                                    else "timeout"
                                ),
                            },
                            "learning_potential": getattr(
                                lifecycle, "acl_learning_potential", lambda *_: None
                            )(
                                int(infos[idx].get("acl_slot_id", idx)),
                                int(infos[idx]["acl_episode_id"]),
                            )
                            if infos[idx].get("acl_episode_id") is not None
                            else None,
                            "info": dict(infos[idx]),
                            "ready_learning_potentials": getattr(
                                lifecycle, "acl_ready_learning_potentials", lambda: {}
                            )(),
                        }
                        for idx in done_indices.tolist()
                    ]
                    if len(done_indices) > 0:
                        lifecycle.on_episode_end(indices=done_indices.tolist())
                        self.preprocessor.reset()
                    event_details: dict[int, tuple[int, str, int, float, float]] = {}
                    for idx in done_indices:
                        if episode_success[idx]:
                            reason = "success"
                        elif episode_collision[idx]:
                            reason = "collision"
                        elif episode_out_of_road[idx]:
                            reason = "out_of_road"
                        else:
                            reason = "timeout"
                        episodes += 1
                        event_details[int(idx)] = (
                            episodes,
                            reason,
                            int(episode_len[idx]),
                            float(episode_scalar_reward[idx]),
                            float(episode_route_completion[idx]),
                        )
                        recent_episode_lens.append(int(episode_len[idx]))
                        recent_episode_rewards.append(float(episode_scalar_reward[idx]))
                        recent_episode_env_rewards.append(float(episode_env_reward[idx]))
                        if episode_has_scalar_rule_reward[idx]:
                            recent_episode_scalar_rule_rewards.append(
                                float(episode_scalar_rule_reward[idx])
                            )
                        if episode_has_hybrid_reward[idx]:
                            recent_episode_hybrid_rewards.append(float(episode_hybrid_reward[idx]))
                        chunk_episode_success.append(1.0 if episode_success[idx] else 0.0)
                        chunk_episode_collision.append(1.0 if episode_collision[idx] else 0.0)
                        chunk_episode_out_of_road.append(1.0 if episode_out_of_road[idx] else 0.0)
                        chunk_episode_route_completion.append(float(episode_route_completion[idx]))
                        episode_len[idx] = 0
                        episode_scalar_reward[idx] = 0.0
                        episode_env_reward[idx] = 0.0
                        episode_scalar_rule_reward[idx] = 0.0
                        episode_hybrid_reward[idx] = 0.0
                        episode_has_scalar_rule_reward[idx] = False
                        episode_has_hybrid_reward[idx] = False
                        episode_success[idx] = False
                        episode_collision[idx] = False
                        episode_out_of_road[idx] = False
                        episode_route_completion[idx] = 0.0

                    if len(done_indices) > 0 and bool(getattr(env, "acl_mode", False)):
                        if vector_episode_end_callback is None:
                            raise RuntimeError(
                                "ACL vector environment requires "
                                "vector_episode_end_callback for parent-controlled reset."
                            )
                        reset_started = time.perf_counter()
                        reset_observations = vector_episode_end_callback(
                            env,
                            done_indices.tolist(),
                            acl_episode_payloads,
                        )
                        phase_seconds["acl_reset_callback"] += time.perf_counter() - reset_started
                        reset_infos = getattr(env, "reset_infos", ())
                        for idx in done_indices.tolist():
                            if idx >= len(reset_infos) or not isinstance(reset_infos[idx], Mapping):
                                continue
                            worker_timing = reset_infos[idx].get(
                                "_thesis_worker_timing_seconds", {}
                            )
                            if isinstance(worker_timing, Mapping):
                                phase_seconds["worker_reset"] += float(worker_timing.get("reset", 0.0))
                            reset_timing = reset_infos[idx].get("_thesis_reset_timing_seconds", {})
                            if isinstance(reset_timing, Mapping):
                                for name, seconds in reset_timing.items():
                                    key = f"reset_{name}"
                                    phase_seconds.setdefault(key, 0.0)
                                    phase_seconds[key] += float(seconds)
                        if set(int(idx) for idx in reset_observations) != set(
                            int(idx) for idx in done_indices.tolist()
                        ):
                            raise ValueError(
                                "ACL vector reset callback must return exactly one observation "
                                "for every completed worker slot."
                            )
                        next_obs = np.asarray(next_obs).copy()
                        for idx, reset_observation in reset_observations.items():
                            next_obs[int(idx)] = reset_observation

                    payload_by_slot = {
                        int(payload["worker_id"]): payload for payload in acl_episode_payloads
                    }
                    for idx in done_indices:
                        payload = payload_by_slot.get(int(idx), {})
                        (
                            event_number,
                            reason,
                            event_length,
                            event_reward,
                            event_route_completion,
                        ) = event_details[int(idx)]
                        context = {
                            key: value
                            for key, value in dict(payload.get("live_event_context", {})).items()
                            if key not in {"slot", "episode_id"}
                        }
                        event_logs.appendleft(
                            _format_episode_event(
                                episode_number=event_number,
                                env_index=idx,
                                episode_length=event_length,
                                reward=event_reward,
                                reason=reason,
                                route_completion=event_route_completion,
                                context=context,
                            )
                        )

                    actor_loss = float(getattr(lifecycle, "last_actor_loss", float("nan")))
                    critic_loss = float(getattr(lifecycle, "last_critic_loss", float("nan")))
                    if not math.isnan(actor_loss):
                        ema_actor_loss = (
                            actor_loss
                            if math.isnan(ema_actor_loss)
                            else (ema_alpha * actor_loss + (1 - ema_alpha) * ema_actor_loss)
                        )
                    if not math.isnan(critic_loss):
                        ema_critic_loss = (
                            critic_loss
                            if math.isnan(ema_critic_loss)
                            else (ema_alpha * critic_loss + (1 - ema_alpha) * ema_critic_loss)
                        )

                    progress.update(progress_task, completed=min(collected_steps, chunk_timesteps))
                    should_render = (
                        log_interval <= 0
                        or collected_steps % log_interval < n_envs
                        or collected_steps >= chunk_timesteps
                    )
                    if should_render:
                        extra_events = (
                            monitor_event_poll_callback() if monitor_event_poll_callback else []
                        )
                        event_logs.extendleft(reversed(extra_events))
                        extra_renderables = (
                            live_extra_renderables_callback()
                            if live_extra_renderables_callback
                            else []
                        )
                        live.update(Group(progress, _table(), *extra_renderables, _logs_panel()))
                    obs = next_obs
                    phase_seconds["logging_callback"] += max(
                        0.0,
                        time.perf_counter() - iteration_started - iteration_accounted_seconds,
                    )
        finally:
            for logger, original_level, original_propagate in monitor_logger_restore_state:
                logger.removeHandler(monitor_log_handler)
                logger.setLevel(original_level)
                logger.propagate = original_propagate

        lifecycle.end_training()
        self.adapter.end_training()
        for name, value in getattr(lifecycle, "update_timing_seconds", {}).items():
            phase_seconds[f"learner_detail_{name}"] = float(value)
        elapsed = max(time.time() - start_time, 1e-9)
        timed_phase_seconds = sum(
            value
            for name, value in phase_seconds.items()
            if not name.startswith(("worker_", "rulebook_", "learner_detail_"))
        )
        phase_seconds["unattributed"] = max(0.0, elapsed - timed_phase_seconds)
        ep_len_final_mean = float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
        ep_len_final_std = float(np.std(recent_episode_lens)) if recent_episode_lens else 0.0
        ep_len_ci_95 = (
            1.96 * ep_len_final_std / np.sqrt(len(recent_episode_lens))
            if len(recent_episode_lens) > 1
            else 0.0
        )
        ep_rew_final_mean = (
            float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0
        )
        ep_rew_final_std = float(np.std(recent_episode_rewards)) if recent_episode_rewards else 0.0
        ep_rew_ci_95 = (
            1.96 * ep_rew_final_std / np.sqrt(len(recent_episode_rewards))
            if len(recent_episode_rewards) > 1
            else 0.0
        )

        return {
            "episodes": int(episodes),
            "ep_len_mean": ep_len_final_mean,
            "ep_len_std": ep_len_final_std,
            "ep_len_ci_95": float(ep_len_ci_95),
            "ep_rew_mean": ep_rew_final_mean,
            "ep_rew_std": ep_rew_final_std,
            "ep_rew_ci_95": float(ep_rew_ci_95),
            "ep_env_rew_mean": float(np.mean(recent_episode_env_rewards))
            if recent_episode_env_rewards
            else None,
            "ep_scalar_rule_rew_mean": float(np.mean(recent_episode_scalar_rule_rewards))
            if recent_episode_scalar_rule_rewards
            else None,
            "ep_hybrid_rew_mean": float(np.mean(recent_episode_hybrid_rewards))
            if recent_episode_hybrid_rewards
            else None,
            "ep_success_rate": float(np.mean(chunk_episode_success))
            if chunk_episode_success
            else None,
            "ep_collision_rate": float(np.mean(chunk_episode_collision))
            if chunk_episode_collision
            else None,
            "ep_out_of_road_rate": float(np.mean(chunk_episode_out_of_road))
            if chunk_episode_out_of_road
            else None,
            "ep_route_completion_mean": float(np.mean(chunk_episode_route_completion))
            if chunk_episode_route_completion
            else None,
            "actor_loss": float(getattr(lifecycle, "last_actor_loss", float("nan"))),
            "critic_loss": float(getattr(lifecycle, "last_critic_loss", float("nan"))),
            "learning_potential": (
                float(np.mean(lifecycle.learning_potential_values))
                if getattr(lifecycle, "learning_potential_values", [])
                else None
            ),
            "actor_loss_ema": float(ema_actor_loss) if not math.isnan(ema_actor_loss) else None,
            "critic_loss_ema": float(ema_critic_loss) if not math.isnan(ema_critic_loss) else None,
            "learning_rate": float(getattr(lifecycle, "last_learning_rate", float("nan"))),
            "update_calls": int(getattr(lifecycle, "update_count", 0)),
            "n_updates": int(getattr(lifecycle, "gradient_step_count", 0)),
            "fps": float(collected_steps / elapsed),
            "elapsed_seconds": float(elapsed),
            "phase_seconds": {name: float(value) for name, value in phase_seconds.items()},
            "chunk_steps_actual": int(collected_steps),
            "train_reset_seed_first": reset_seeds_used[0] if reset_seeds_used else None,
            "train_reset_seed_last": reset_seeds_used[-1] if reset_seeds_used else None,
            "train_reset_seed_unique_count": len(set(reset_seeds_used)),
            "last_observations": obs,
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
        show_progress: bool = True,
        artifact_recorder_factory: Any | None = None,
        before_episode_reset_callback: Callable[[Any, int], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        progress_description: str = "Evaluation episodes",
    ) -> dict[str, Any]:
        """Evaluate the agent in the given environment for a specified number of episodes.
        Args:
            env: The environment to evaluate in. Must have `reset()` and `step()` methods.
            n_eval_episodes: Number of episodes to evaluate for.
            deterministic: Whether to use deterministic actions during evaluation.
            before_episode_reset_callback: Optional callback invoked immediately
                before each episode reset.
        Returns:
            Aggregate metrics for the full evaluation set. If `return_episode_metrics=True`,
            the output also includes a `per_episode` section with raw episode vectors.
        """
        # Validate `n_eval_episodes`
        if int(n_eval_episodes) <= 0:
            raise ValueError("`n_eval_episodes` must be > 0.")
        if count_envs(env) > 1:
            return self._evaluate_parallel(
                env=env,
                n_eval_episodes=int(n_eval_episodes),
                deterministic=deterministic,
                base_seed=base_seed,
                return_episode_metrics=return_episode_metrics,
                error_priority_base=error_priority_base,
                show_progress=show_progress,
                artifact_recorder_factory=artifact_recorder_factory,
                before_episode_reset_callback=before_episode_reset_callback,
                progress_callback=progress_callback,
                progress_description=progress_description,
            )

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
        episode_hybrid_returns: list[float | None] = []
        episode_rule_rewards_by_rule: list[dict[str, float]] = []
        episode_video_paths: list[str | None] = []
        episode_video_authoritative_paths: list[str | None] = []
        episode_video_manifest_paths: list[str | None] = []
        episode_trajectory_log_paths: list[str | None] = []
        episode_video_recorded_live: list[bool] = []
        episode_replay_warnings: list[str | None] = []
        episode_scenario_metadata: list[dict[str, Any]] = []
        all_rule_names: set[str] = set()
        rule_priority_by_name: dict[str, int] = {}
        per_rule_episode_min_margins: dict[str, list[float]] = {}
        per_rule_violation_count: dict[str, int] = {}

        progress: Progress | None = None
        progress_task: Any | None = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.start()
            progress_task = progress.add_task(
                f"{progress_description} (0/{n_eval_episodes})",
                total=n_eval_episodes,
            )

        try:
            # Loop over evaluation episodes
            for episode_idx in range(n_eval_episodes):
                # Reset preprocessor and environment state at the start of each episode
                self.preprocessor.reset()
                if before_episode_reset_callback is not None:
                    before_episode_reset_callback(env, episode_idx)
                scenario_seed = int(base_seed) + episode_idx if base_seed is not None else None
                if base_seed is not None:
                    obs, reset_info = env.reset(seed=scenario_seed)
                else:
                    obs, reset_info = env.reset()
                metadata_keys = (
                    "scenario_uid",
                    "scenario_id",
                    "source",
                    "split",
                    "arm",
                    "scenario_arm",
                    "worker_id",
                    "sampling_mode",
                    "requested_arm",
                    "source_cell_fallback",
                )
                episode_metadata = {
                    key: reset_info.get(key)
                    for key in metadata_keys
                    if isinstance(reset_info, dict) and key in reset_info
                }
                artifact_recorder = (
                    artifact_recorder_factory(
                        {
                            "episode_idx": episode_idx,
                            "episode_id": episode_idx + 1,
                            "scenario_seed": scenario_seed,
                            "deterministic": deterministic,
                        }
                    )
                    if artifact_recorder_factory is not None
                    else None
                )

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
                ep_hybrid_return = 0.0
                ep_has_hybrid_reward = False

                # Loop until episode ends
                while not (done or truncated):
                    # Get action
                    current_obs = obs
                    action, _ = self.predict(current_obs, deterministic=deterministic)
                    next_obs, scalar_reward, done, truncated, step_info = env.step(action)
                    obs = next_obs
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
                        hybrid_reward = step_info.get("hybrid_reward")
                        if isinstance(hybrid_reward, (int, float, np.floating)):
                            ep_hybrid_return += float(hybrid_reward)
                            ep_has_hybrid_reward = True
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
                    ep_route_completion = max(
                        ep_route_completion, self._extract_route_completion(step_info)
                    )
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
                    if artifact_recorder is not None:
                        artifact_recorder.record_step(
                            env=env,
                            step_index=ep_step_count - 1,
                            observation=current_obs,
                            next_observation=next_obs,
                            action=np.asarray(action, dtype=np.float32),
                            reward=float(scalar_reward),
                            done=bool(done),
                            truncated=bool(truncated),
                            step_info=step_info,
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
                episode_hybrid_returns.append(
                    float(ep_hybrid_return) if ep_has_hybrid_reward else None
                )
                episode_rule_rewards_by_rule.append(
                    dict(sorted(ep_rule_reward_sum_by_rule.items()))
                )
                if ep_step_count > 0:
                    episode_top_rule_violation_rate.append(
                        ep_top_rule_violating_steps / ep_step_count
                    )
                else:
                    episode_top_rule_violation_rate.append(0.0)

                violated_in_episode: list[tuple[str, int]] = []
                for rule_name, min_margin in ep_rule_min_margin.items():
                    per_rule_episode_min_margins.setdefault(rule_name, []).append(float(min_margin))
                    if float(min_margin) < 0.0:
                        per_rule_violation_count[rule_name] = int(
                            per_rule_violation_count.get(rule_name, 0) + 1
                        )
                        violated_in_episode.append(
                            (rule_name, int(rule_priority_by_name.get(rule_name, 0)))
                        )

                if violated_in_episode:
                    violated_in_episode.sort(key=lambda item: (item[1], item[0]))
                    violation_pattern = "+".join(name for name, _prio in violated_in_episode)
                    episode_violated_rule_names.append(violation_pattern)
                else:
                    violation_pattern = "none"
                    episode_violated_rule_names.append("none")
                episode_violation_patterns.append(violation_pattern)

                if ep_rule_min_margin:
                    p_max = max(
                        int(rule_priority_by_name.get(name, 0)) for name in ep_rule_min_margin
                    )
                    ev_episode = 0.0
                    for rule_name, min_margin in ep_rule_min_margin.items():
                        priority = int(rule_priority_by_name.get(rule_name, 0))
                        weight = float(error_priority_base) ** float(p_max - priority)
                        ev_episode += weight * max(0.0, -float(min_margin))
                    episode_error_values.append(float(ev_episode))
                else:
                    episode_error_values.append(0.0)

                episode_metrics = {
                    "reward": float(ep_return),
                    "env_reward": float(ep_env_return),
                    "scalar_rule_reward": float(ep_scalar_rule_return)
                    if ep_has_scalar_rule_reward
                    else None,
                    "hybrid_reward": float(ep_hybrid_return) if ep_has_hybrid_reward else None,
                    "episode_length": int(ep_step_count),
                    "success": bool(ep_success),
                    "collision": bool(ep_collision),
                    "out_of_road": bool(ep_out_of_road),
                    "timeout": bool(truncated),
                    "route_completion": float(ep_route_completion),
                    "top_rule_violation_rate": float(episode_top_rule_violation_rate[-1]),
                    "error_value": float(episode_error_values[-1]),
                    "violated_rules": episode_violated_rule_names[-1],
                    "violation_pattern": violation_pattern,
                }
                artifact_payload = (
                    artifact_recorder.finalize_episode(episode_metrics=episode_metrics)
                    if artifact_recorder is not None
                    else {}
                )
                episode_video_paths.append(artifact_payload.get("video_path"))
                episode_video_authoritative_paths.append(
                    artifact_payload.get("video_authoritative_path")
                )
                episode_video_manifest_paths.append(artifact_payload.get("video_manifest_path"))
                episode_trajectory_log_paths.append(artifact_payload.get("trajectory_log_path"))
                episode_video_recorded_live.append(
                    bool(artifact_payload.get("video_recorded_live", False))
                )
                episode_replay_warnings.append(artifact_payload.get("replay_warning"))
                if isinstance(step_info, dict):
                    episode_metadata.update(
                        {key: step_info[key] for key in metadata_keys if key in step_info}
                    )
                    episode_metadata["termination_reason"] = step_info.get("termination_reason")
                episode_metadata["terminated"] = bool(done)
                episode_metadata["truncated"] = bool(truncated)
                episode_scenario_metadata.append(episode_metadata)

                if progress is not None and progress_task is not None:
                    progress.advance(progress_task)
                if progress_callback is not None:
                    progress_callback(episode_idx + 1, n_eval_episodes)
                if progress is not None and progress_task is not None:
                    progress.update(
                        progress_task,
                        description=f"{progress_description} ({episode_idx + 1}/{n_eval_episodes})",
                    )
        finally:
            if progress is not None:
                progress.stop()

        counterexample_rate = (
            float(
                np.mean(
                    [1.0 if pattern != "none" else 0.0 for pattern in episode_violation_patterns]
                )
            )
            if episode_violation_patterns
            else 0.0
        )
        violated_rules_ratio = float(
            sum(1 for rule in all_rule_names if int(per_rule_violation_count.get(rule, 0)) > 0)
        ) / float(max(len(all_rule_names), 1))
        unique_violation_patterns = int(len(set(episode_violation_patterns)))

        per_rule_rows: list[dict[str, Any]] = []
        for rule_name in sorted(
            all_rule_names, key=lambda name: (rule_priority_by_name.get(name, 0), name)
        ):
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

        scalar_rule_available_values = [
            value for value in episode_scalar_rule_returns if value is not None
        ]
        hybrid_available_values = [value for value in episode_hybrid_returns if value is not None]

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
            "mean_hybrid_reward": (
                float(np.mean(hybrid_available_values)) if hybrid_available_values else None
            ),
            "std_hybrid_reward": (
                float(np.std(hybrid_available_values)) if hybrid_available_values else None
            ),
            "mean_rule_saturation_max": float(np.mean(episode_saturation_max)),
            "collision_rate": float(np.mean(episode_collision)),
            "collision_rate_std": float(np.std(episode_collision)),
            "out_of_road_rate": float(np.mean(episode_out_of_road)),
            "success_rate": float(np.mean(episode_success)),
            "success_rate_std": float(np.std(episode_success)),
            "route_completion": float(np.mean(episode_route_completion)),
            "top_rule_violation_rate": float(np.mean(episode_top_rule_violation_rate)),
            "avg_error_value": float(np.mean(episode_error_values))
            if episode_error_values
            else 0.0,
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
                "hybrid_returns": episode_hybrid_returns,
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
                "video_path": episode_video_paths,
                "video_authoritative_path": episode_video_authoritative_paths,
                "video_manifest_path": episode_video_manifest_paths,
                "trajectory_log_path": episode_trajectory_log_paths,
                "video_recorded_live": episode_video_recorded_live,
                "replay_warning": episode_replay_warnings,
                "scenario_metadata": episode_scenario_metadata,
            }
        return metrics

    def _evaluate_parallel(
        self,
        *,
        env: Any,
        n_eval_episodes: int,
        deterministic: bool,
        base_seed: int | None,
        return_episode_metrics: bool,
        error_priority_base: float,
        show_progress: bool,
        artifact_recorder_factory: Any | None,
        before_episode_reset_callback: Callable[[Any, int], None] | None,
        progress_callback: Callable[[int, int], None] | None,
        progress_description: str,
    ) -> dict[str, Any]:
        """Evaluate active episodes concurrently with deterministic reduction."""

        if not all(callable(getattr(env, name, None)) for name in ("step_slots", "reset_slots")):
            raise TypeError("Parallel evaluation requires the repository vector environment.")
        if type(self.preprocessor).__name__ != "IdentityPreprocessor":
            raise ValueError(
                "Parallel evaluation currently requires the stateless IdentityPreprocessor."
            )

        worker_count = int(getattr(env, "num_envs", 0))
        if worker_count <= 1:
            raise ValueError("Parallel evaluation requires at least two workers.")
        scenario_indices = getattr(env, "evaluation_scenario_indices", None)
        if scenario_indices is not None and len(scenario_indices) != n_eval_episodes:
            raise ValueError("Evaluation scenario index count does not match episode count.")

        def _scenario_seed(episode_idx: int) -> int | None:
            if base_seed is None:
                return None
            return int(base_seed) + int(episode_idx)

        def _reset_seed(episode_idx: int) -> int | None:
            if base_seed is not None:
                return _scenario_seed(episode_idx)
            if scenario_indices is not None:
                return int(scenario_indices[episode_idx])
            return None

        def _render_kwargs(recorder: Any) -> dict[str, Any]:
            cfg = getattr(recorder, "_topdown_cfg", {})
            return {
                "mode": "topdown",
                "window": bool(cfg.get("window", False)),
                "screen_record": bool(cfg.get("screen_record", False)),
                "screen_size": tuple(cfg.get("screen_size", [800, 800])),
                "scaling": float(cfg.get("scaling", 4)),
                "semantic_map": bool(cfg.get("semantic_map", False)),
                "diagnostic_geometry": True,
            }

        self.preprocessor.reset()
        records: list[dict[str, Any] | None] = [None] * n_eval_episodes
        active: dict[int, int] = {}
        observations: dict[int, Any] = {}
        states: dict[int, _ParallelEvaluationEpisode] = {}

        def _install_episode(slot: int, episode_idx: int) -> None:
            proxy = env.get_slot_proxy(slot)
            if before_episode_reset_callback is not None:
                before_episode_reset_callback(proxy, episode_idx)
            scenario_seed = _scenario_seed(episode_idx)
            recorder = (
                artifact_recorder_factory(
                    {
                        "episode_idx": episode_idx,
                        "episode_id": episode_idx + 1,
                        "scenario_seed": scenario_seed,
                        "deterministic": deterministic,
                    }
                )
                if artifact_recorder_factory is not None
                else None
            )
            active[slot] = episode_idx
            states[slot] = _ParallelEvaluationEpisode(
                episode_idx=episode_idx,
                scenario_seed=scenario_seed,
                reset_info={},
                artifact_recorder=recorder,
            )

        initial_count = min(worker_count, n_eval_episodes)
        for slot in range(initial_count):
            _install_episode(slot, slot)
        initial_resets = env.reset_slots(
            list(range(initial_count)),
            seeds={slot: _reset_seed(slot) for slot in range(initial_count)},
        )
        for slot in range(initial_count):
            observations[slot], reset_info = initial_resets[slot]
            state = states[slot]
            state.metadata = {
                key: reset_info.get(key)
                for key in state._METADATA_KEYS
                if isinstance(reset_info, dict) and key in reset_info
            }

        progress: Progress | None = None
        progress_task: Any | None = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.start()
            progress_task = progress.add_task(
                f"{progress_description} (0/{n_eval_episodes})",
                total=n_eval_episodes,
            )

        next_episode = initial_count
        try:
            while active:
                actions: dict[int, Any] = {}
                for slot in sorted(active):
                    action, _ = self.predict(observations[slot], deterministic=deterministic)
                    actions[slot] = action
                step_results = env.step_slots(actions)

                render_slots = [
                    slot for slot in sorted(active) if states[slot].artifact_recorder is not None
                ]
                if render_slots:
                    recorder = states[render_slots[0]].artifact_recorder
                    frames = env.render_slots(_render_kwargs(recorder), slots=render_slots)
                    for slot in render_slots:
                        env.get_slot_proxy(slot).set_rendered_frame(frames[slot])

                completed_slots: list[int] = []
                for slot in sorted(active):
                    next_obs, reward, done, info, _reset_info = step_results[slot]
                    done_flag = bool(done)
                    truncated = bool(
                        isinstance(info, dict)
                        and info.get("truncated", info.get("TimeLimit.truncated", False))
                    )
                    terminated = bool(
                        isinstance(info, dict)
                        and info.get("terminated", done_flag and not truncated)
                    )
                    state = states[slot]
                    state.observe_step(
                        self,
                        env=env.get_slot_proxy(slot),
                        observation=observations[slot],
                        next_observation=next_obs,
                        action=actions[slot],
                        scalar_reward=reward,
                        done=terminated,
                        truncated=truncated,
                        step_info=info,
                    )
                    observations[slot] = next_obs
                    if done_flag:
                        episode_idx = active[slot]
                        records[episode_idx] = state.finalize(
                            self,
                            done=terminated,
                            truncated=truncated,
                            error_priority_base=error_priority_base,
                        )
                        completed_slots.append(slot)
                        del active[slot]
                        del states[slot]
                        observations.pop(slot, None)
                        if next_episode < n_eval_episodes:
                            _install_episode(slot, next_episode)
                            next_episode += 1

                if completed_slots:
                    reset_slots = [slot for slot in completed_slots if slot in active]
                    if reset_slots:
                        reset_results = env.reset_slots(
                            reset_slots,
                            seeds={slot: _reset_seed(active[slot]) for slot in reset_slots},
                        )
                        for slot in reset_slots:
                            observations[slot], reset_info = reset_results[slot]
                            state = states[slot]
                            state.metadata = {
                                key: reset_info.get(key)
                                for key in state._METADATA_KEYS
                                if isinstance(reset_info, dict) and key in reset_info
                            }
                    if progress is not None and progress_task is not None:
                        progress.update(progress_task, advance=len(completed_slots))
                    if progress_callback is not None:
                        completed_count = n_eval_episodes - len(active)
                        progress_callback(completed_count, n_eval_episodes)
                    if progress is not None and progress_task is not None:
                        completed_count = n_eval_episodes - len(active)
                        progress.update(
                            progress_task,
                            description=f"{progress_description} ({completed_count}/{n_eval_episodes})",
                        )
        finally:
            if progress is not None:
                progress.stop()

        if any(record is None for record in records):
            raise RuntimeError("Parallel evaluation ended without one result per episode.")
        return self._aggregate_parallel_evaluation(
            [record for record in records if record is not None],
            error_priority_base=error_priority_base,
            return_episode_metrics=return_episode_metrics,
        )

    def _aggregate_parallel_evaluation(
        self,
        records: list[dict[str, Any]],
        *,
        error_priority_base: float,
        return_episode_metrics: bool,
    ) -> dict[str, Any]:
        """Reduce ordered slot results with the sequential metric definitions."""

        records = sorted(records, key=lambda record: int(record["episode_idx"]))
        all_rule_names: set[str] = set()
        rule_priority_by_name: dict[str, int] = {}
        per_rule_episode_min_margins: dict[str, list[float]] = {}
        per_rule_violation_count: dict[str, int] = {}
        for record in records:
            priorities = record.get("rule_priorities", {})
            margins = record.get("rule_min_margins", {})
            for name, priority in priorities.items():
                all_rule_names.add(str(name))
                if str(name) not in rule_priority_by_name:
                    rule_priority_by_name[str(name)] = int(priority)
            for name, margin in margins.items():
                name = str(name)
                all_rule_names.add(name)
                per_rule_episode_min_margins.setdefault(name, []).append(float(margin))
                if float(margin) < 0.0:
                    per_rule_violation_count[name] = int(per_rule_violation_count.get(name, 0) + 1)

        episode_returns = [float(record["reward"]) for record in records]
        episode_env_returns = [float(record["env_reward"]) for record in records]
        episode_scalar_rule_returns = [record.get("scalar_rule_reward") for record in records]
        episode_hybrid_returns = [record.get("hybrid_reward") for record in records]
        scalar_available = [value for value in episode_scalar_rule_returns if value is not None]
        hybrid_available = [value for value in episode_hybrid_returns if value is not None]
        episode_saturation_max = [float(record.get("saturation_max", 0.0)) for record in records]
        episode_collision = [float(record["collision"]) for record in records]
        episode_out_of_road = [float(record["out_of_road"]) for record in records]
        episode_success = [float(record["success"]) for record in records]
        episode_route_completion = [float(record["route_completion"]) for record in records]
        episode_top_rule_rate = [float(record["top_rule_violation_rate"]) for record in records]
        episode_lengths = [int(record["episode_length"]) for record in records]
        episode_timeout = [float(record["timeout"]) for record in records]
        episode_error_values = [float(record["error_value"]) for record in records]
        episode_patterns = [str(record["violation_pattern"]) for record in records]
        episode_violated = [str(record["violated_rules"]) for record in records]

        per_rule_rows: list[dict[str, Any]] = []
        for rule_name in sorted(
            all_rule_names, key=lambda name: (rule_priority_by_name.get(name, 0), name)
        ):
            margins = per_rule_episode_min_margins.get(rule_name, [])
            violation_count = int(per_rule_violation_count.get(rule_name, 0))
            per_rule_rows.append(
                {
                    "rule_name": rule_name,
                    "rule_priority": int(rule_priority_by_name.get(rule_name, 0)),
                    "violated": bool(violation_count > 0),
                    "violation_rate": float(violation_count / max(len(records), 1)),
                    "violation_count": violation_count,
                    "mean_margin": float(np.mean(margins)) if margins else 0.0,
                    "min_margin": float(np.min(margins)) if margins else 0.0,
                    "max_margin": float(np.max(margins)) if margins else 0.0,
                }
            )

        metrics: dict[str, Any] = {
            "mean_reward": float(np.mean(episode_returns)),
            "std_reward": float(np.std(episode_returns)),
            "mean_env_reward": float(np.mean(episode_env_returns)),
            "std_env_reward": float(np.std(episode_env_returns)),
            "mean_scalar_rule_reward": float(np.mean(scalar_available))
            if scalar_available
            else None,
            "std_scalar_rule_reward": float(np.std(scalar_available)) if scalar_available else None,
            "mean_hybrid_reward": float(np.mean(hybrid_available)) if hybrid_available else None,
            "std_hybrid_reward": float(np.std(hybrid_available)) if hybrid_available else None,
            "mean_rule_saturation_max": float(np.mean(episode_saturation_max)),
            "collision_rate": float(np.mean(episode_collision)),
            "collision_rate_std": float(np.std(episode_collision)),
            "out_of_road_rate": float(np.mean(episode_out_of_road)),
            "success_rate": float(np.mean(episode_success)),
            "success_rate_std": float(np.std(episode_success)),
            "route_completion": float(np.mean(episode_route_completion)),
            "top_rule_violation_rate": float(np.mean(episode_top_rule_rate)),
            "avg_error_value": float(np.mean(episode_error_values)),
            "max_error_value": float(np.max(episode_error_values)),
            "counterexample_rate": float(
                np.mean([1.0 if pattern != "none" else 0.0 for pattern in episode_patterns])
            ),
            "violated_rules_ratio": float(
                sum(1 for name in all_rule_names if per_rule_violation_count.get(name, 0) > 0)
            )
            / float(max(len(all_rule_names), 1)),
            "unique_violation_patterns": int(len(set(episode_patterns))),
            "per_rule": per_rule_rows,
        }
        if return_episode_metrics:
            metrics["per_episode"] = {
                "returns": episode_returns,
                "env_returns": episode_env_returns,
                "scalar_rule_returns": episode_scalar_rule_returns,
                "hybrid_returns": episode_hybrid_returns,
                "rule_rewards_by_rule": [record["rule_rewards_by_rule"] for record in records],
                "saturation_max": episode_saturation_max,
                "collision": episode_collision,
                "out_of_road": episode_out_of_road,
                "success": episode_success,
                "route_completion": episode_route_completion,
                "top_rule_violation_rate": episode_top_rule_rate,
                "episode_length": episode_lengths,
                "timeout": episode_timeout,
                "error_value": episode_error_values,
                "violation_pattern": episode_patterns,
                "violated_rules": episode_violated,
                "video_path": [record.get("video_path") for record in records],
                "video_authoritative_path": [
                    record.get("video_authoritative_path") for record in records
                ],
                "video_manifest_path": [record.get("video_manifest_path") for record in records],
                "trajectory_log_path": [record.get("trajectory_log_path") for record in records],
                "video_recorded_live": [
                    bool(record.get("video_recorded_live", False)) for record in records
                ],
                "replay_warning": [record.get("replay_warning") for record in records],
                "scenario_metadata": [record["scenario_metadata"] for record in records],
            }
        return metrics

    @staticmethod
    def _extract_rule_margins(step_info: Any) -> list[tuple[str, int, float]]:
        if not isinstance(step_info, dict):
            return []
        names = (
            step_info.get("rule_metadata", {}).get("rule_names")
            if isinstance(step_info.get("rule_metadata"), dict)
            else None
        )
        priorities = (
            step_info.get("rule_metadata", {}).get("priorities")
            if isinstance(step_info.get("rule_metadata"), dict)
            else None
        )
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
        """Save planner and adapter state to the specified checkpoint path."""
        # Save planner state
        self.planner.save(checkpoint_path)
        # Save adapter state if the adapter is trainable
        if bool(getattr(self.adapter, "requires_training", False)):
            adapter_ckpt = self.adapter_checkpoint_path(checkpoint_path)
            self.adapter.save(str(adapter_ckpt))
        if self.checkpoint_identity is not None:
            write_reward_semantics_sidecar(checkpoint_path, self.checkpoint_identity)

    def save_generation(
        self,
        checkpoint_root: str | Path,
        checkpoint_name: str,
        manifest: CheckpointManifest,
    ) -> CheckpointGeneration:
        """Publish planner state as an immutable, manifest-validated generation.

        The explicit method keeps historical ``save(.zip)`` behavior available
        for legacy experiments while giving approved v1.1 runs a fail-closed
        checkpoint contract.
        """

        generation = publish_checkpoint_generation(
            checkpoint_root=checkpoint_root,
            checkpoint_name=checkpoint_name,
            save_model=lambda model_path: self.planner.save(model_path),
            manifest=manifest,
        )
        if bool(getattr(self.adapter, "requires_training", False)):
            adapter_path = generation.generation_dir / "adapter.pt"
            self.adapter.save(str(adapter_path))
        if self.checkpoint_identity is not None:
            write_reward_semantics_sidecar(
                generation.model_path,
                self.checkpoint_identity,
            )
        return generation

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
