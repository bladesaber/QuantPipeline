from typing import Literal
import torch
import torch.nn as nn
import numpy as np

class EnvSetting(object):
    def __init__(
        self,
        train_env_num: int,
        test_env_num: int,
        vector_env_method: Literal['sequential', 'parallel', 'cluster'],
        train_buffer_size: int, 
        test_buffer_size: int,
        ignore_obs_next: bool = False, 
        stack_num: int = 1, 
        is_prioritized: bool = False, 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        weight_norm: bool = True,
        train_exploration_noise: bool = True,
        test_exploration_noise: bool = False,
    ):
        self.train_env_num = train_env_num
        self.test_env_num = test_env_num
        self.vector_env_method = vector_env_method
        self.train_buffer_size = train_buffer_size
        self.test_buffer_size = test_buffer_size
        self.ignore_obs_next = ignore_obs_next
        self.stack_num = stack_num
        self.is_prioritized = is_prioritized
        self.alpha = alpha
        self.beta = beta
        self.weight_norm = weight_norm
        self.train_exploration_noise = train_exploration_noise
        self.test_exploration_noise = test_exploration_noise


class PolicySetting(object):
    def __init__(
        self,
        optim_style: Literal['adam', 'sgd'],
        lr: float,
        gamma: float = 0.99,
        action_bound_method: Literal['', 'clip', 'tanh'] = '',
        action_scaling: bool = False,
        lr_scheduler: Literal['ReduceLROnPlateau', 'StepLR'] = '',
    ):
        self.optim_style = optim_style
        self.lr = lr
        self.gamma = gamma
        self.action_bound_method = action_bound_method
        self.action_scaling = action_scaling
        self.lr_scheduler = lr_scheduler


class DQNPolicySetting(PolicySetting):
    def __init__(self, n_step: int, target_update_freq: int, is_double: bool, clip_loss_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad


class PPOPolicySetting(PolicySetting):
    def __init__(
        self, 
        eps_clip=0.2, 
        dual_clip=None,
        value_clip=False,
        advantage_normalization=True,
        recompute_advantage=False,
        vf_coef=0.5, 
        ent_coef=0.01, 
        max_grad_norm=0.5, 
        gae_lambda=0.98, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.advantage_normalization = advantage_normalization
        self.recompute_advantage = recompute_advantage
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

class SACPolicySetting(PolicySetting):
    def __init__(
        self, 
        tau=0.05, 
        alpha=0.2, 
        estimation_step=1, 
        reward_normalization=False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.alpha = alpha
        self.estimation_step = estimation_step
        self.reward_normalization = reward_normalization
