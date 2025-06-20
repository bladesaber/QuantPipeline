import os
import shutil
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import time
import tensorboardX

import gymnasium as gym
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.algorithms.algorithm import Algorithm


class RlEpisodeUtils(object):
    @staticmethod
    def get_env_from_episode(episode: SingleAgentEpisode | MultiAgentEpisode, indexs: list[int] = None):
        return episode.get_observations(indices=indexs)

    @staticmethod
    def get_rl_module_from_episode(episode: SingleAgentEpisode | MultiAgentEpisode, indexs: list[int] = None):
        return episode.get_actions(indices=indexs)

    @staticmethod
    def length(episode: SingleAgentEpisode | MultiAgentEpisode):
        return len(episode)

    @staticmethod
    def get_info_from_episode(episode: SingleAgentEpisode | MultiAgentEpisode, indexs: list[int] = None):
        return episode.get_infos(indices=indexs)

    @staticmethod
    def is_done(episode: SingleAgentEpisode | MultiAgentEpisode):
        return episode.is_done


class TensorboardCallback(RLlibCallback):
    loss_keys = ["total_loss"]
    
    def __init__(
        self, 
        log_dir: str, 
        log_episode_tasks: list[str] = ["reward", "length"], 
        log_train_tasks: list[str] = ["time_cost", "loss"],
        train_loss_keys: list[str] = None
    ):
        if not os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)
        self.writer = tensorboardX.SummaryWriter(log_dir)
        
        self.log_episode_tasks = log_episode_tasks
        self.log_train_tasks = log_train_tasks
        if train_loss_keys is None:
            self.train_loss_keys = self.loss_keys
        else:
            self.train_loss_keys = train_loss_keys
        
        self.episode_count = 0
        self.episode_done_num = 0
    
    def log_episode_data(self, episode: SingleAgentEpisode | MultiAgentEpisode, indices: list[int] = None):
        if "reward" in self.log_episode_tasks:
            rewards: np.ndarray = episode.get_rewards(indices=indices)
            self.writer.add_scalar("episode/reward", np.sum(rewards), time.time())
        if "length" in self.log_episode_tasks:
            self.writer.add_scalar("episode/length", len(episode), time.time())
        if episode.is_done:
            self.episode_done_num += 1
        self.episode_count += 1

    def log_train_data(self, metrics_logger: MetricsLogger):
        result: dict = metrics_logger.peek()
        
        if 'time_cost' in self.log_train_tasks:
            self.writer.add_scalar("time_cost/train", result['timers']["training_step"], time.time())
            self.writer.add_scalar("time_cost/env_sample", result['timers']["env_runner_sampling_timer"], time.time())
            self.writer.add_scalar("time_cost/synch_weights", result['timers']["synch_weights"], time.time())
            self.writer.add_scalar("time_cost/inference", result['env_runners']["rlmodule_inference_timer"], time.time())

        for key in self.train_loss_keys:
            self.writer.add_scalar(f"loss/{key}", result['learners']['default_policy'][key], time.time())

    def on_episode_end(
        self,
        *,
        episode: SingleAgentEpisode | MultiAgentEpisode,
        prev_episode_chunks = None,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env = None,
        policies = None,
        **kwargs,
    ):
        self.log_episode_data(episode)
    
    def on_episode_start(self, *, episode: SingleAgentEpisode | MultiAgentEpisode, **kwargs):
        # episode.custom_data["custom_data"] = {}
        pass
    
    def on_episode_step(self, *, episode: SingleAgentEpisode | MultiAgentEpisode, **kwargs):
        # episode.custom_data["custom_data"]["obs_list"].append(episode.get_observations())
        pass
    
    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
        result: dict,
        **kwargs,
    ):
        self.log_train_data(metrics_logger)
    
    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
        result: dict,
        **kwargs,
    ):
        self.log_eval_data(result)
    
    
    