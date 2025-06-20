import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Literal
from abc import ABC, abstractmethod

from gym_env import GymEnv


class AccountStat(object):
    def __init__(
        self, 
        return_n_step: int = 1, 
        return_aggregate: Literal['mean', 'last'] = 'last', 
        positive_risk_aversion: float = 0.5, 
        negative_risk_aversion: float = 0.1, 
        operation_loss: float = 0.01,
        **kwargs
    ):
        self.prev_portfolio = 0.0
        self.cur_portfolio = 0.0
        
        self.return_n_step = return_n_step
        self.return_aggregate = return_aggregate
        self.positive_risk_aversion = positive_risk_aversion
        self.negative_risk_aversion = negative_risk_aversion
        self.operation_loss = operation_loss
    
    @staticmethod
    def isoelastic_utility(rewards: np.ndarray | float, risk_aversion: float) -> float:
        if risk_aversion == 1.0:
            return np.log(rewards)
        else:
            return (rewards ** (1 - risk_aversion) - 1) / (1 - risk_aversion)
    
    @staticmethod
    def compute_exp_utility(reward: float, risk_aversion: float = 0.1) -> float:
        return np.exp(abs(reward) * risk_aversion) * reward
    
    def compute_return_utility(self, next_portfolio: float, prev_price: float, future_prices: np.ndarray) -> float:
        """
        range of next_portfolio is -1.0 ~ 1.0, range of reward is -1.0 ~ 1.0
        :param positive_risk_aversion: should be 0.0 ~ 2.0
        :param negative_risk_aversion: should be 0.0 ~ 0.5
        """
        if self.return_aggregate == 'last' or self.return_n_step == 1:
            reward = next_portfolio * (future_prices[-1] / prev_price - 1.0)
        elif self.return_aggregate == 'mean':
            reward = next_portfolio * (future_prices[:self.return_n_step].mean() / prev_price - 1.0)
        else:
            raise ValueError(f"aggregate must be one of 'mean', 'last', but got {self.return_aggregate}")

        if self.positive_risk_aversion > 0 and reward >= 0:
            return AccountStat.isoelastic_utility(reward + 1.0, self.positive_risk_aversion)
        if self.negative_risk_aversion > 0 and reward < 0:
            return AccountStat.compute_exp_utility(reward, self.negative_risk_aversion)
        
        return reward
    
    def compute_operation_utility(self, next_portfolio: float, prev_portfolio: float) -> float:
        """only penalize operation when portfolio changes"""
        if next_portfolio != prev_portfolio:
            return self.operation_loss
        else:
            return 0.0

    def compute_reward(self, future_prices: np.ndarray, prev_price: float) -> float:
        utility = self.compute_return_utility(self.cur_portfolio, prev_price, future_prices)
        utility += self.compute_operation_utility(self.cur_portfolio, self.prev_portfolio)
        return utility

    def update_stat(self, portfolio: float):
        self.prev_portfolio = self.cur_portfolio
        self.cur_portfolio = portfolio

    def reset(self):
        self.prev_portfolio = 0.0
        self.cur_portfolio = 0.0


class SingleAssetEnv(GymEnv, ABC):
    """Only consider profit and loss without considering cash/asset limit"""
    def __init__(
        self, 
        is_continuous: bool, 
        action_dim: int, 
        direction: Literal['long', 'short', 'both'],
        cool_down_step: int = 10,
        return_n_step: int = 1,
        return_aggregate: Literal['mean', 'last'] = 'last',
        positive_risk_aversion: float = 0.5,
        negative_risk_aversion: float = 0.1,
        operation_loss: float = 0.01,
        **kwargs
    ):
        env_config = {}
        if is_continuous:
            env_config['action_dim'] = 1
        else:
            if direction == 'both':
                env_config['action_dim'] = 2 * action_dim + 1
                env_config['action_start'] = -action_dim
            else:
                env_config['action_dim'] = action_dim + 1

            if direction == 'long':
                env_config['action_start'] = 0
            elif direction == 'short':
                env_config['action_start'] = -action_dim
        
        if direction == 'both':
            env_config['action_low'] = -1.0
            env_config['action_high'] = 1.0
        elif direction == 'long':
            env_config['action_low'] = 0.0
            env_config['action_high'] = 1.0
        elif direction == 'short':
            env_config['action_low'] = -1.0
            env_config['action_high'] = 0.0
        
        self.account_stat = AccountStat(
            return_n_step=return_n_step,
            return_aggregate=return_aggregate,
            positive_risk_aversion=positive_risk_aversion,
            negative_risk_aversion=negative_risk_aversion,
            operation_loss=operation_loss,
            **kwargs
        )
        self.cool_down_step = cool_down_step
        self.market_data: pd.DataFrame = None
        self.trade_obs: pd.DataFrame = None
        self.stamp_iloc: int = self.cool_down_step

        self._prepare_data(**kwargs)
        env_config['observation_shape'] = self.trade_obs.shape[1:]
        
        super().__init__(env_config)
        self.action_scale: float = None
        if self.is_action_discrete:
            self.action_scale = 1 / (self.action_dim - 1)

    @abstractmethod
    def _simulate_market(self, step: int, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Market observation generation must be implemented")

    @abstractmethod
    def _market2obs(self, market_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """convert market data to machine learning features observation"""
        raise NotImplementedError("Market observation generation must be implemented")
    
    def _prepare_data(self, **kwargs) -> None:
        self.market_data = self._simulate_market(**kwargs)
        self.trade_obs = self._market2obs(self.market_data, **kwargs)
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple[np.ndarray, dict]:
        self._prepare_data(**kwargs)
        self.account_stat.reset()
        self.stamp_iloc = self.cool_down_step
        
        obs = self.trade_obs.iloc[self.stamp_iloc]
        self.stamp_iloc += 1
        return obs, {}

    def step(self, action: float | int, **kwargs) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.is_action_discrete:
            next_portfolio = action * self.action_scale
        else:
            assert abs(action) <= 1.0, f"action must be in range of [-1.0, 1.0], but got {action}"
            next_portfolio = action
        self.account_stat.update_stat(next_portfolio)
        
        next_obs: pd.DataFrame = self.trade_obs.iloc[self.stamp_iloc: self.stamp_iloc + self.account_stat.return_n_step]
        prev_obs: pd.Series = self.trade_obs.iloc[self.stamp_iloc - 1]
        reward = self.account_stat.compute_reward(next_obs.price.values, prev_obs.price)
        
        if self.stamp_iloc == len(self.trade_obs) - 1 - self.account_stat.return_n_step:
            return next_obs, reward, True, False, {}
        else:
            self.stamp_iloc += 1
        
        return next_obs, reward, False, False, {}

    def render(self):
        raise NotImplementedError("Render is not implemented")

