import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class TsBaseNetwork(nn.Module, ABC):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.setup()
    
    @abstractmethod
    def setup(self):
        raise NotImplementedError("Please implement the setup method")
        
    @abstractmethod
    def forward(self, obs, state=None, info={}):
        """please return logits(torch.Tensor), state(torch.Tensor)"""
        raise NotImplementedError("Please implement the forward method")


class EasyTsNetwork(TsBaseNetwork):
    def setup(self):
        self.model = nn.Sequential(
            nn.Linear(self.state_shape, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, self.action_shape),
        )
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        logits = self.model(obs)
        return logits, state
