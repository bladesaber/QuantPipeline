from typing import Type
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from functools import partial

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_distributions import TorchDistribution, TorchCategorical, TorchDiagGaussian
from ray.rllib.algorithms.bc.torch.default_bc_torch_rl_module import DefaultBCTorchRLModule


class TorchCategoricalWithTemp(TorchCategorical):
    def __init__(self, logits=None, probs=None, temperature: float = 1.0):
        """Initializes a TorchCategoricalWithTemp instance.
        Args:
            logits: Event log probabilities (non-normalized).
            probs: The probabilities of each event.
            temperature: In case of using logits, this parameter can be used to
                determine the sharpness of the distribution. i.e.
                ``probs = softmax(logits / temperature)``. The temperature must be
                strictly positive. A low value (e.g. 1e-10) will result in argmax
                sampling while a larger value will result in uniform sampling.
        """
        assert (temperature > 0.0), f"Temperature ({temperature}) must be strictly positive!"
        if logits is not None:
            logits /= temperature
        else:
            probs = torch.nn.functional.softmax(probs / temperature)
        super().__init__(logits, probs)


class RlNetworkModule(TorchRLModule, ABC):
    """
        TorchRLModule derived from RLModule
            encoder: convert observation to embedding
            hidden layer: process embedding
            pi head: compute action logits from embedding
    """
    def __init__(
        self, 
        observation_space: gym.Space, action_space: gym.Space, model_config: dict, 
        inference_only: bool = False, learner_only: bool = False, **kwargs
    ):
        self.input_dim = observation_space.shape[0]
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_action_discrete = True
            self.output_dim = action_space.n
        else:
            self.is_action_discrete = False
            self.output_dim = action_space.shape[0]
        self.exploration_temp = model_config.get("exploration_temp", -1.0)
        
        # since super().__init__ will call setup(), please initialize input_dim and output_dim before super().__init__
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            inference_only=inference_only,
            learner_only=learner_only,
        )

    @abstractmethod
    def setup(self):
        raise NotImplementedError('Please implement setup() in your subclass')
    
    @abstractmethod
    def _forward_train(self, batch: dict) -> dict:
        """Please only return action logits"""
        raise NotImplementedError('Please implement _forward_train() in your subclass')
    
    @abstractmethod
    def _forward_inference(self, batch: dict) -> dict:
        """Please return action logits and actions"""
        raise NotImplementedError('Please implement _forward_inference() in your subclass')
    
    def _forward_exploration(self, batch: dict) -> dict:
        """Please only return actions"""
        raise NotImplementedError('Please implement _forward_exploration() in your subclass')

    def get_exploration_action_dist_cls(self, *args, **kwargs) -> Type[TorchDistribution]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            if self.exploration_temp > 0.0:
                return partial(TorchCategoricalWithTemp, temperature=self.exploration_temp)
            else:
                return TorchCategorical
        elif isinstance(self.action_space, gym.spaces.Box):
            return TorchDiagGaussian
        else:
            raise ValueError("[ERROR] Unsupported action space")
        

class EasyRlNetworkModule(RlNetworkModule):
    """Easy example of RlNetworkModule"""
    def __init__(
        self, 
        observation_space: gym.Space, action_space: gym.Space, model_config: dict, 
        inference_only: bool = False, learner_only: bool = False, **kwargs
    ):
        self.epsilon = model_config.get("epsilon", 0.0)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            inference_only=inference_only,
            learner_only=learner_only,
            **kwargs
        )
    
    def setup(self):
        self._encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU()
        )
        self._hidden_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self._pi_head = nn.Sequential(
            nn.Linear(128, self.output_dim),
            nn.Tanh()
        )
    
    def _forward_train(self, batch: dict) -> dict:
        embedding = self._encoder_net(batch[Columns.OBS])
        hidden = self._hidden_layer(embedding)
        action_logits = self._pi_head(hidden)
        return { Columns.ACTION_DIST_INPUTS: action_logits }
    
    def _forward_inference(self, batch: dict) -> dict:
        with torch.no_grad():
            embedding = self._encoder_net(batch[Columns.OBS])
            hidden = self._hidden_layer(embedding)
            action_logits = self._pi_head(hidden)
            return {
                Columns.ACTIONS: torch.argmax(action_logits, dim=1), 
                Columns.ACTION_DIST_INPUTS: action_logits
            }
    
    def _forward_exploration(self, batch: dict) -> dict:
        with torch.no_grad():
            embedding = self._encoder_net(batch[Columns.OBS])
            hidden = self._hidden_layer(embedding)
            action_logits = self._pi_head(hidden)
            return { Columns.ACTION_DIST_INPUTS: action_logits }


class DefaultNetworkModule(object):
    fcnet_config = {
        "fcnet_hiddens": [128, 128],  # two layers of 128 hidden units
        "fcnet_activation": "relu",
    }
    
    conv_config = {
        "conv_filters": [
            [16, [3, 3], 2], # 1st CNN layer: num_filters, kernel, stride(, padding)?
            [32, [3, 3], 2],
            [64, [3, 3], 2],
            [128, [3, 3], 2],
        ],
        "conv_activation": "relu",
        # After the last CNN, the default model flattens, then adds an optional MLP.
        "head_fcnet_hiddens": [256],
        "head_fcnet_activation": "relu",
    }
    
    @staticmethod
    def get_mlp_default_model_config() -> dict:
        return DefaultNetworkModule.fcnet_config
    
    @staticmethod
    def get_cnn_default_model_config() -> dict:
        return DefaultNetworkModule.conv_config

    @staticmethod
    def get_mlp_default_model_spec(observation_space: gym.Space, action_space: gym.Space) -> RLModule:
        return DefaultBCTorchRLModule(
            observation_space=observation_space,
            action_space=action_space,
            model_config=DefaultNetworkModule.get_mlp_default_model_config(),
        )

    @staticmethod
    def get_cnn_default_model_spec(observation_space: gym.Space, action_space: gym.Space) -> RLModule:
        return DefaultBCTorchRLModule(
            observation_space=observation_space,
            action_space=action_space,
            model_config=DefaultNetworkModule.get_cnn_default_model_config(),
        )
