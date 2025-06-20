import os
import shutil
import numpy as np
import pandas as pd
from typing import List, Callable, Literal

import gymnasium as gym
import tianshou as ts
from tianshou.data import Batch
from tianshou.env import BaseVectorEnv
from tianshou.data import Collector
from tianshou.data import ReplayBuffer, ReplayBufferManager
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger


class TrainCallback(object):
    def __init__(
        self,
        log_dir: str,
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        repeat_per_collect: int = 1,
    ):
        """
        :param int max_epoch: the maximum number of epochs to train
        :param int step_per_epoch: the number of environment step (a.k.a. transition) collected per epoch
        :param int step_per_collect: the number of transition the collector would collect before the network update. For example, 
            the code above means “collect 10 transitions and do one policy network update”
        :param int episode_per_test: the number of episodes for one policy evaluation
        :param int batch_size: the number of transitions sampled from the replay buffer for one policy network update
        :param int repeat_per_collect: the number of repeat time for policy learning, for example, set it to 2 means the policy needs to learn each given batch data twice
        """
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect
        self.episode_per_test = episode_per_test
        self.batch_size = batch_size
        self.repeat_per_collect = repeat_per_collect
        
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.logger = TensorboardLogger(self.writer)
        
    def train_fn(self, epoch: int, env_step: int) -> None:
        return None

    def test_fn(self, epoch: int, env_step: int) -> None:
        return None
    
    def stop_fn(self, mean_rewards: float) -> bool:
        return False


class TsUtils(object):
    @staticmethod
    def buffer2np(buffer: ReplayBuffer | ReplayBufferManager, index: List[int]) -> np.ndarray:
        return {
            'obs': buffer.obs[index],
            'act': buffer.act[index],
            'rew': buffer.rew[index],
            'terminated': buffer.terminated[index],
            'truncated': buffer.truncated[index],
            'done': buffer.done[index],
            # 'obs_next': buffer.obs_next[index]
        }
    
    @staticmethod
    def buffer_sample(buffer: ReplayBuffer | ReplayBufferManager, batch_size: int, is_index: bool = False) -> np.ndarray | Batch:
        if is_index:
            return buffer.sample_indices(batch_size)
        else:
            return buffer.sample(batch_size)

    @staticmethod
    def buffer_next_indexs(buffer: ReplayBuffer | ReplayBufferManager, indexs: List[int]) -> List[int]:
        """if meet the end of the buffer, return the last index"""
        return buffer.next(indexs)
    
    @staticmethod
    def buffer_prev_indexs(buffer: ReplayBuffer | ReplayBufferManager, indexs: List[int]) -> List[int]:
        """if meet the start of the buffer, return the first index"""
        return buffer.prev(indexs)
    
    @staticmethod
    def collect_n_steps(collector: Collector, n_step: int, random: bool = False) -> dict:
        """random: whether to use random policy for collecting data"""
        stats = collector.collect(n_step=n_step, random=random)
        return stats
    
    @staticmethod
    def collect_n_episodes(collector: Collector, n_episode: int, random: bool = False) -> dict:
        """random: whether to use random policy for collecting data"""
        stats = collector.collect(n_episode=n_episode, random=random)
        return stats

    @staticmethod
    def collector_reset_envs(collector: Collector) -> None:
        collector.reset()

    @staticmethod
    def create_vector_envs(env_class: Callable, env_num: int, method: Literal['sequential', 'parallel', 'cluster']) -> BaseVectorEnv:
        """example: env_class = lambda: gym.make('CartPole-v1'), env_num = 4, method = 'parallel'"""
        if method == 'sequential':
            return ts.env.DummyVectorEnv([env_class for _ in range(env_num)])
        elif method == 'parallel':
            return ts.env.SubprocVectorEnv([env_class for _ in range(env_num)])
        elif method == 'cluster':
            return ts.env.RayVectorEnv([env_class for _ in range(env_num)])
        else:
            raise ValueError(f"Invalid method: {method}")

    @staticmethod
    def create_buffer(
        env_obj: BaseVectorEnv | gym.Env, 
        buffer_size: int, ignore_obs_next: bool = False, stack_num = 1, 
        is_prioritized: bool = False, alpha: float = 0.6, beta: float = 0.4, weight_norm: bool = True
    ) -> ReplayBuffer | ReplayBufferManager:
        """
        :param int stack_num: the frame-stack sampling argument, should be greater than or equal to 1. Default to 1 (no stacking).
        if env_obj is BaseVectorEnv, then the total_size will be split into env_num, like second env start from buffer_size * 1, third env start from buffer_size * 2, etc.
        """
        if isinstance(env_obj, BaseVectorEnv):
            if is_prioritized:
                return ts.data.PrioritizedVectorReplayBuffer(
                    total_size=buffer_size, buffer_num=env_obj.env_num, stack_num=stack_num, alpha=alpha, beta=beta, weight_norm=weight_norm,
                    ignore_obs_next=ignore_obs_next
                )
            else:
                return ts.data.VectorReplayBuffer(
                    total_size=buffer_size, buffer_num=env_obj.env_num, stack_num=stack_num, ignore_obs_next=ignore_obs_next
                )
        else:
            if is_prioritized:
                return ts.data.PrioritizedReplayBuffer(
                    total_size=buffer_size, stack_num=stack_num, alpha=alpha, beta=beta, weight_norm=weight_norm,
                    ignore_obs_next=ignore_obs_next
                )
            else:
                return ts.data.ReplayBuffer(
                    total_size=buffer_size, stack_num=stack_num, ignore_obs_next=ignore_obs_next
                )
    
    

