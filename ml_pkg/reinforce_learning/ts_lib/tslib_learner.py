import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from typing import Literal, Callable
from torch.distributions import Categorical, Normal, Independent, Distribution

import gymnasium as gym
import tianshou as ts

from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

# ------ easy to use networks ------
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.continuous import Actor, Critic
# from tianshou.utils.net.discrete import Actor, Critic
# ------ easy to use networks ------

from tianshou.policy import PPOPolicy, DQNPolicy, SACPolicy, TD3Policy
from tianshou.trainer import BaseTrainer

from ts_lib.tslib_network import TsBaseNetwork
from ts_lib.tslib_utils import TsUtils, TrainCallback
from ts_lib.ts_setting import EnvSetting
from ts_lib.ts_setting import PolicySetting, DQNPolicySetting, PPOPolicySetting, SACPolicySetting

"""
Collector
    |- Policy
    |
    |- EnvObject
    |    |- Env
    |    |- VectorEnv
    |
    |- Buffer
    |    |- ReplayBuffer
    |    |- PrioritizedReplayBuffer
    
"""

def discrete_dist_fn(logits: torch.Tensor) -> Distribution:
    return Categorical(logits=logits)

def continuous_dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)


class DQNLearner(object):
    def __init__(
        self, 
        model: TsBaseNetwork, 
        env_class: Callable, 
        env_setting: EnvSetting, 
        policy_setting: PolicySetting, 
        train_callback: TrainCallback
    ):
        self.model = model
        self.env_class: Callable = env_class
        
        _env_tmp = env_class()
        self.observation_space = _env_tmp.observation_space
        self.action_space = _env_tmp.action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.is_action_discrete = True
        elif isinstance(self.action_space, gym.spaces.Box):
            self.is_action_discrete = False
        else:
            raise ValueError(f"Invalid action space: {self.action_space}")
        
        self.env_setting = env_setting
        self.train_env_obj: BaseVectorEnv | gym.Env = None
        self.test_env_obj: BaseVectorEnv | gym.Env = None
        self.train_collector: ts.data.Collector = None
        self.test_collector: ts.data.Collector = None
        self.policy: BasePolicy = None
        self.policy_setting: PolicySetting | DQNPolicySetting = policy_setting
        self.optim: torch.optim.Optimizer = None
        self.trainer: BaseTrainer = None
        self.train_callback: TrainCallback = train_callback

    def setup_envs(self):
        if self.env_setting.train_env_num > 1:
            self.train_env_obj = TsUtils.create_vector_envs(self.env_class, self.env_setting.train_env_num, self.env_setting.vector_env_method)
        else:
            self.train_env_obj = self.env_class()
        
        if self.env_setting.test_env_num > 1:
            self.test_env_obj = TsUtils.create_vector_envs(self.env_class, self.env_setting.test_env_num, self.env_setting.vector_env_method)
        else:
            self.test_env_obj = self.env_class()

    def setup_policy(self):
        self.optim = self.setup_optim(self.model, self.policy_setting.lr, self.policy_setting.optim_style)
        self.policy = DQNPolicy(
            model=self.model,
            optim=self.optim,
            
            discount_factor=self.policy_setting.gamma,
            estimation_step=self.policy_setting.n_step,
            target_update_freq=self.policy_setting.target_update_freq,
            is_double=self.policy_setting.is_double,
            clip_loss_grad=self.policy_setting.clip_loss_grad,
            
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_scheduler=self.setup_lr_scheduler(self.optim, self.policy_setting.lr_scheduler),
            action_bound_method=self.policy_setting.action_bound_method,
            action_scaling=self.policy_setting.action_scaling,
        )

    def setup_collector(self):
        self.train_collector = ts.data.Collector(
            policy=self.policy, 
            env=self.train_env_obj, 
            buffer=TsUtils.create_buffer(
                env_obj=self.train_env_obj, 
                buffer_size=self.env_setting.train_buffer_size, 
                ignore_obs_next=self.env_setting.ignore_obs_next, 
                stack_num=self.env_setting.stack_num, 
                is_prioritized=self.env_setting.is_prioritized, 
                alpha=self.env_setting.alpha, 
                beta=self.env_setting.beta, 
                weight_norm=self.env_setting.weight_norm
            ), 
            exploration_noise=self.policy_setting.train_exploration_noise
        )
        self.test_collector = ts.data.Collector(
            policy=self.policy, 
            env=self.test_env_obj, 
            buffer=TsUtils.create_buffer(
                env_obj=self.test_env_obj, 
                buffer_size=self.env_setting.test_buffer_size, 
                ignore_obs_next=self.env_setting.ignore_obs_next, 
                stack_num=self.env_setting.stack_num, 
                is_prioritized=False
            ), 
            exploration_noise=self.policy_setting.test_exploration_noise
        )

    def setup(self):
        self.setup_envs()
        self.setup_policy()
        self.setup_collector()
        self.trainer = ts.trainer.OffpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.train_callback.max_epoch, 
            step_per_epoch=self.train_callback.step_per_epoch, 
            step_per_collect=self.train_callback.step_per_collect,
            episode_per_test=self.train_callback.episode_per_test, 
            batch_size=self.train_callback.batch_size,
            train_fn=self.train_callback.train_fn,
            test_fn=self.train_callback.test_fn,
            stop_fn=self.train_callback.stop_fn,
            verbose=True,
            show_progress=True,
            test_in_train=True,
            logger=self.train_callback.logger
        )
    
    def train(self) -> dict:
        return self.trainer.run()
    
    @staticmethod
    def setup_optim(model: TsBaseNetwork, lr: float = 1e-3, optim_style: Literal['adam', 'sgd'] = 'adam', **kwargs):
        if optim_style == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optim_style == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=kwargs.get('momentum', 0.9))
        else:
            raise ValueError(f"Invalid optimizer style: {optim_style}")

    @staticmethod
    def setup_lr_scheduler(
        optim: torch.optim.Optimizer, lr_scheduler: Literal['ReduceLROnPlateau', 'StepLR'] = '', **kwargs
    ):
        if lr_scheduler == '':
            return None
        elif lr_scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 
                factor=kwargs.get('factor', 0.1), 
                patience=kwargs.get('patience', 100),
                threshold_mode=kwargs.get('threshold_mode', 'rel'),
                cooldown=kwargs.get('cooldown', 100),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif lr_scheduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                optim, 
                step_size=kwargs.get('step_size', 10), 
                gamma=kwargs.get('gamma', 0.1),
                last_epoch=kwargs.get('last_epoch', -1)
            )
        else:
            raise ValueError(f"Invalid lr scheduler: {lr_scheduler}")
    
    def save_state(self, path: str):
        torch.save(self.policy.state_dict(), path)
    
    def load_state(self, path: str):
        self.policy.load_state_dict(torch.load(path))
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))


class PpoLearner(DQNLearner):
    def __init__(
        self,
        actor: TsBaseNetwork, 
        critic: TsBaseNetwork, 
        env_class: Callable, 
        env_setting: EnvSetting, 
        policy_setting: PolicySetting, 
        train_callback: TrainCallback
    ):
        self.actor = actor
        self.critic = critic
        self.env_class: Callable = env_class
        
        _env_tmp = env_class()
        self.observation_space = _env_tmp.observation_space
        self.action_space = _env_tmp.action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.is_action_discrete = True
        elif isinstance(self.action_space, gym.spaces.Box):
            self.is_action_discrete = False
        else:
            raise ValueError(f"Invalid action space: {self.action_space}")
        
        self.env_setting = env_setting
        self.train_env_obj: BaseVectorEnv | gym.Env = None
        self.test_env_obj: BaseVectorEnv | gym.Env = None
        self.train_collector: ts.data.Collector = None
        self.test_collector: ts.data.Collector = None
        self.policy: BasePolicy = None
        self.policy_setting: PolicySetting | PPOPolicySetting = policy_setting
        self.optim: torch.optim.Optimizer = None
        self.trainer: BaseTrainer = None
        self.train_callback: TrainCallback = train_callback
    
    def setup_policy(self):
        self.optim = self.setup_optim(self.model, self.policy_setting.lr, self.policy_setting.optim_style)
        self.policy = PPOPolicy(
            actor=self.actor,
            critic=self.critic,
            optim=self.optim,
            
            discount_factor=self.policy_setting.gamma,
            dist_fn=discrete_dist_fn if self.is_action_discrete else continuous_dist_fn,
            eps_clip=self.policy_setting.eps_clip,
            dual_clip=self.policy_setting.dual_clip,
            value_clip=self.policy_setting.value_clip,
            advantage_normalization=self.policy_setting.advantage_normalization,
            recompute_advantage=self.policy_setting.recompute_advantage,
            vf_coef=self.policy_setting.vf_coef,
            ent_coef=self.policy_setting.ent_coef,
            max_grad_norm=self.policy_setting.max_grad_norm,
            gae_lambda=self.policy_setting.gae_lambda,
            
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_scheduler=self.setup_lr_scheduler(self.optim, self.policy_setting.lr_scheduler),
            action_bound_method=self.policy_setting.action_bound_method,
            action_scaling=self.policy_setting.action_scaling,
        )

    def setup(self):
        self.setup_envs()
        self.setup_policy()
        self.setup_collector()
        self.trainer = ts.trainer.OnpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.train_callback.max_epoch, 
            step_per_epoch=self.train_callback.step_per_epoch, 
            step_per_collect=self.train_callback.step_per_collect,
            repeat_per_collect=self.train_callback.repeat_per_collect,
            episode_per_test=self.train_callback.episode_per_test, 
            batch_size=self.train_callback.batch_size,
            train_fn=self.train_callback.train_fn,
            test_fn=self.train_callback.test_fn,
            stop_fn=self.train_callback.stop_fn,
            verbose=True,
            show_progress=True,
            test_in_train=True,
            logger=self.train_callback.logger
        )
        

class SacLearner(PpoLearner):
    def __init__(
        self,
        actor: TsBaseNetwork, 
        critic: TsBaseNetwork, 
        critic_2: TsBaseNetwork,
        env_class: Callable, 
        env_setting: EnvSetting, 
        policy_setting: PolicySetting | SACPolicySetting, 
        train_callback: TrainCallback
    ):
        super().__init__(actor, critic, env_class, env_setting, policy_setting, train_callback)
        self.critic_2 = critic_2
            
    def setup_policy(self):
        self.actor_optim = self.setup_optim(self.actor, self.policy_setting.lr, self.policy_setting.optim_style)
        self.critic_optim = self.setup_optim(self.critic, self.policy_setting.lr, self.policy_setting.optim_style)
        self.critic_2_optim = self.setup_optim(self.critic_2, self.policy_setting.lr, self.policy_setting.optim_style)
        
        self.policy = SACPolicy(
            actor=self.actor,
            critic1=self.critic,
            critic2=self.critic_2,
            
            actor_optim=self.actor_optim,
            critic1_optim=self.critic_optim,
            critic2_optim=self.critic_2_optim,
            
            gamma=self.policy_setting.gamma,
            tau=self.policy_setting.tau,
            alpha=self.policy_setting.alpha,
            estimation_step=self.policy_setting.estimation_step,
            reward_normalization=self.policy_setting.reward_normalization,
            
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_scheduler=self.setup_lr_scheduler(self.optim, self.policy_setting.lr_scheduler),
            action_bound_method=self.policy_setting.action_bound_method,
            action_scaling=self.policy_setting.action_scaling,
        )

    def setup(self):
        self.setup_envs()
        self.setup_policy()
        self.setup_collector()
        self.trainer = ts.trainer.OffpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.train_callback.max_epoch, 
            step_per_epoch=self.train_callback.step_per_epoch, 
            step_per_collect=self.train_callback.step_per_collect,
            episode_per_test=self.train_callback.episode_per_test, 
            batch_size=self.train_callback.batch_size,
            train_fn=self.train_callback.train_fn,
            test_fn=self.train_callback.test_fn,
            stop_fn=self.train_callback.stop_fn,
            verbose=True,
            show_progress=True,
            test_in_train=True,
            logger=self.train_callback.logger
        )

