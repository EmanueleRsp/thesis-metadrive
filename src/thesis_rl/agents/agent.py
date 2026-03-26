from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from thesis_rl.agents.base import BasePlannerBackend


class Agent:
    """Composable agent that applies preprocessor -> planner -> adapter."""

    def __init__(self, preprocessor: Any, planner: BasePlannerBackend, adapter: Any) -> None:
        self.preprocessor = preprocessor
        self.planner = planner
        self.adapter = adapter

    def train(
        self,
        env: Any,
        total_timesteps: int,
        deterministic: bool = False,
        log_interval: int = 1000,
    ) -> None:
        lifecycle = self.planner.get_lifecycle()
        lifecycle.begin_training(total_timesteps=total_timesteps)
        self.adapter.begin_training()

        self.preprocessor.reset()
        obs, _ = env.reset()

        start_time = time.time()
        episodes = 0
        episode_len = 0
        episode_reward = 0.0
        recent_episode_lens: deque[int] = deque(maxlen=100)
        recent_episode_rewards: deque[float] = deque(maxlen=100)

        for step in range(1, total_timesteps + 1):
            processed_obs = self.preprocessor(obs)
            planner_output = lifecycle.act(processed_obs, deterministic=deterministic)
            action = self.adapter(planner_output)

            next_obs, reward, done, truncated, step_info = env.step(action)
            terminated = bool(done or truncated)
            episode_len += 1
            episode_reward += float(reward)

            next_processed_obs = self.preprocessor(next_obs)
            lifecycle.observe_transition(
                observation=processed_obs,
                action=action,
                reward=float(reward),
                done=terminated,
                next_observation=next_processed_obs,
            )
            lifecycle.maybe_update()
            self.adapter.maybe_update()

            if terminated:
                episodes += 1
                recent_episode_lens.append(episode_len)
                recent_episode_rewards.append(episode_reward)

                episode_len = 0
                episode_reward = 0.0
                self.preprocessor.reset()
                obs, _ = env.reset()
            else:
                obs = next_obs

            if log_interval > 0 and step % log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                fps = int(step / elapsed)
                ep_len_mean = float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0
                ep_rew_mean = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0

                actor_loss = getattr(lifecycle, "last_actor_loss", float("nan"))
                critic_loss = getattr(lifecycle, "last_critic_loss", float("nan"))
                learning_rate = getattr(lifecycle, "last_learning_rate", float("nan"))
                n_updates = int(getattr(lifecycle, "update_count", 0))

                print("---------------------------------")
                print("| rollout/           |\t\t|")
                print(f"|    ep_len_mean     | {ep_len_mean:.2f}\t|")
                print(f"|    ep_rew_mean     | {ep_rew_mean:.2f}\t|")
                print("| time/              |\t\t|")
                print(f"|    episodes        | {episodes}\t|")
                print(f"|    fps             | {fps}\t|")
                print(f"|    time_elapsed    | {int(elapsed)}\t|")
                print(f"|    total_timesteps | {step}\t|")
                print("| train/             |\t\t|")
                print(f"|    actor_loss      | {actor_loss:.3g}\t|")
                print(f"|    critic_loss     | {critic_loss:.3g}\t|")
                print(f"|    learning_rate   | {learning_rate:.3g}\t|")
                print(f"|    n_updates       | {n_updates}\t|")
                sat_summary = self._extract_saturation_summary(step_info)
                if sat_summary is not None:
                    sat_rule, sat_ratio = sat_summary
                    print(f"|    rho_sat_max     | {sat_ratio:.3f}\t|")
                    print(f"|    rho_sat_rule    | {sat_rule}\t|")
                print("---------------------------------")

        lifecycle.end_training()
        self.adapter.end_training()

    def predict(self, observation: Any, deterministic: bool = False):
        processed_obs = self.preprocessor(observation)
        planner_output, state = self.planner.predict(processed_obs, deterministic=deterministic)
        action = self.adapter(planner_output)
        return action, state

    def evaluate(self, env: Any, n_eval_episodes: int, deterministic: bool = False) -> dict[str, float]:
        episode_returns: list[float] = []
        episode_saturation_max: list[float] = []
        episode_collision: list[float] = []
        episode_out_of_road: list[float] = []
        episode_success: list[float] = []
        episode_route_completion: list[float] = []
        episode_top_rule_violation_rate: list[float] = []

        for _ in range(n_eval_episodes):
            self.preprocessor.reset()
            obs, _ = env.reset()
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

            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, step_info = env.step(action)
                ep_return += float(reward)
                ep_step_count += 1

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

            episode_returns.append(ep_return)
            episode_saturation_max.append(ep_sat_max)
            episode_collision.append(1.0 if ep_collision else 0.0)
            episode_out_of_road.append(1.0 if ep_out_of_road else 0.0)
            episode_success.append(1.0 if ep_success else 0.0)
            episode_route_completion.append(ep_route_completion)
            if ep_step_count > 0:
                episode_top_rule_violation_rate.append(ep_top_rule_violating_steps / ep_step_count)
            else:
                episode_top_rule_violation_rate.append(0.0)

        return {
            "mean_reward": float(np.mean(episode_returns)),
            "std_reward": float(np.std(episode_returns)),
            "mean_rule_saturation_max": float(np.mean(episode_saturation_max)),
            "collision_rate": float(np.mean(episode_collision)),
            "out_of_road_rate": float(np.mean(episode_out_of_road)),
            "success_rate": float(np.mean(episode_success)),
            "route_completion": float(np.mean(episode_route_completion)),
            "top_rule_violation_rate": float(np.mean(episode_top_rule_violation_rate)),
        }

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
        self.planner.save(checkpoint_path)
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
