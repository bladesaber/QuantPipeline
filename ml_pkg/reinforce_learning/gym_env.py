from abc import ABC, abstractmethod
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

import ray
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class GymEnv(gym.Env, ABC):
    def __init__(self, env_config: dict):
        super().__init__()
        self.env_config = env_config
        self.observation_shape = env_config["observation_shape"]
        self.action_dim = env_config["action_dim"]
        self.is_action_discrete = self.action_dim == 1
        if self.is_action_discrete:
            self.action_space = spaces.Discrete(
                self.action_dim, start=self.env_config.get("action_start", 0)
            )
        else:
            self.action_space = spaces.Box(
                low=self.env_config.get("action_low", -np.inf), 
                high=self.env_config.get("action_high", np.inf), 
                shape=(self.action_dim,), dtype=np.float32
            )
        self.observation_space = spaces.Box(
            low=self.env_config.get("obs_low", -np.inf), 
            high=self.env_config.get("obs_high", np.inf), 
            shape=self.observation_shape, dtype=np.float32
        )

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset the environment, return the initial observation and info"""
        raise NotImplementedError("Reset method must be implemented")
    
    @abstractmethod
    def step(self, action: float | int, **kwargs) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment, return the next observation, reward, done, truncated, and info"""
        raise NotImplementedError("Step method must be implemented")

    @abstractmethod
    def render(self):
        """Render the environment"""
        raise NotImplementedError("Render method must be implemented")
    
    def simulate_by_random_actions(self, num_episodes: Optional[int] = None, delay: float = 0.1):
        num_iter = 0
        obs, _ = self.reset()
        done = False
        while not done:
            self.render()
            action = self.action_space.sample()
            obs, reward, done, truncated, info = self.step(action)

            time.sleep(delay)
            num_iter += 1
            
            if done or truncated or (num_episodes is not None and num_iter >= num_episodes):
                break
    
    @staticmethod
    def parallel_simulate_ray(algorithm_config: AlgorithmConfig, is_single_agent: bool, num_episodes_per_worker: int, num_workers: int):
        if is_single_agent:
            env_runners = [ray.remote(SingleAgentEnvRunner).remote(config=algorithm_config) for _ in range(num_workers)]
        else:
            env_runners = [ray.remote(MultiAgentEnvRunner).remote(config=algorithm_config) for _ in range(num_workers)]
        episodes = ray.get([
            worker.sample.remote(num_episodes=num_episodes_per_worker) for worker in env_runners
        ])
        return episodes
    