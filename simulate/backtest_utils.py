import pandas as pd
import numpy as np
from typing import Literal


class MultiPeriodBars(object):
    """Multi period for only one asset"""
    def __init__(self):
        self.main_period: str = None
        self.main_bar: pd.DataFrame = None
        
        self.period_stamps: list[tuple[pd.Timestamp, dict[str, int]]] = []
        self.aux_bars: dict[str, pd.DataFrame] = {}
        
        self.timestamp: pd.Timestamp = None
        self.scence_bars: dict[str, pd.DataFrame] = {}

    def add_main_period_bar(self, period: str, bar: pd.DataFrame):
        MultiPeriodBars.validate_period_bar(period, bar)
        self.main_period = period
        self.main_bar = bar
        for iloc, timestamp in enumerate(bar.index):
            self.period_stamps.append((timestamp, {}))
        
    def add_aux_period_bar(
        self, 
        period: str, other_bar: pd.DataFrame,
        tolerate_days: int = 0, tolerate_hours: int = 0, tolerate_minutes: int = 0, tolerate_seconds: int = 0,
    ):
        MultiPeriodBars.validate_period_bar(period, other_bar)
        assert self.main_bar is not None, f'Main period bar is not set'
        
        self.aux_bars[period] = other_bar
        
        main_ilocs = self.main_bar.index.searchsorted(other_bar.index, side='left')
        
        # todo: remove this assert in production
        assert np.all(self.main_bar.index[main_ilocs] >= other_bar.index), \
            f'Main bar index must be greater than other bar index'
        
        timestamp_diff = (self.main_bar.index[main_ilocs] - other_bar.index).total_seconds()
        masks = timestamp_diff >= (
            tolerate_days * 24 * 60 * 60 
            + tolerate_hours * 60 * 60 
            + tolerate_minutes * 60 
            + tolerate_seconds
        )
        for other_iloc, (mask, main_iloc) in enumerate(zip(masks, main_ilocs)):
            if mask:
                self.period_stamps[main_iloc][period] = other_iloc
    
    @staticmethod
    def validate_period_bar(period: str, bar: pd.DataFrame):
        assert isinstance(bar.index, pd.Timestamp), f'Period {period} bars index must be a timestamp'
        assert bar.index.is_unique, f'Period {period} bars index must be unique'
        assert bar.index.is_monotonic_increasing, f'Period {period} bars index must be sorted'

    def update_scence(self, iloc: int, backward_window: int):
        self.timestamp = self.period_stamps[iloc][0]
        self.scence_bars.clear()
        self.scence_bars[self.main_period] = self.main_bar.iloc[iloc - backward_window:iloc]
        for period, other_iloc in self.period_stamps[iloc][1].items():
            self.scence_bars[period] = self.aux_bars[period].iloc[other_iloc - backward_window:other_iloc]

    @property
    def __len__(self):
        return len(self.period_stamps)
    
    @property
    def periods(self):
        return list(self.aux_bars.keys()) + [self.main_period]
    
    def get_main_bar(self, iloc: int):
        if self.main_bar is None:
            raise ValueError('Main bar is not set')
        return self.main_bar.iloc[iloc]

    def get_timestamp(self, iloc: int):
        return self.period_stamps[iloc][0]


class OrderInfo(object):
    """ Order的自动管理与被动管理不能同时存在 """
    execute_status = ['alive', 'finished', 'failed']
    
    def __init__(
        self, 
        order_id: int,
        date_time: pd.Timestamp, 
        order_direction: Literal['long', 'short'],
        order_style: Literal['open', 'close', 'none'],
        price: float = None, volume: float = 0,
        stop_loss_price: float = None, take_profit_price: float = None,
        close_date_time: pd.Timestamp = None,
    ):
        self.order_id = order_id
        self.date_time = date_time
        self.order_direction = order_direction
        self.order_style = order_style
        
        self.price = price
        self.volume = volume
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.close_date_time = close_date_time
        
        self.msg = ''
        self.status = OrderInfo.execute_status[0]
        self.profit = 0.0
        
        self.slippage_cost = 0.0
        self.context_cost = 0.0
        self.deposit = 0.0

        self.is_triggered_tp = False
        self.is_triggered_sl = False


class AccountStatistic(object):
    def __init__(self, asset_inital_value: float):
        self.asset_inital_value = asset_inital_value
        self.cash_value: float = self.asset_inital_value
        self.long_asset_value: float = 0
        self.short_asset_value: float = 0
        self.long_volume: float = 0
        self.short_volume: float = 0
        self.avg_long_price: float = 0
        self.avg_short_price: float = 0
        
    def reset(self):
        self.cash_value = self.asset_inital_value
        self.long_asset_value = 0
        self.short_asset_value = 0
        self.long_volume = 0
        self.short_volume = 0
        self.avg_long_price = 0
        self.avg_short_price = 0
    
    def to_dict(self):
        return {
            'cash_value': self.cash_value,
            'long_asset_value': self.long_asset_value,
            'short_asset_value': self.short_asset_value,
            'long_volume': self.long_volume,
            'short_volume': self.short_volume,
            'avg_long_price': self.avg_long_price,
            'avg_short_price': self.avg_short_price,
        }

    def open_order(self, order: OrderInfo) -> tuple[bool, str]:
        require_cash = order.deposit + order.slippage_cost + order.context_cost
        if self.cash_value < require_cash:
            return False, '[Error]: Not enough asset value'
        
        self.cash_value -= require_cash
        if order.order_direction == 'long':
            self.long_asset_value += order.deposit
            self.avg_long_price = (
                self.avg_long_price * self.long_volume + order.price * order.volume
            ) / (self.long_volume + order.volume)
            self.long_volume += order.volume
        else:
            self.short_asset_value += order.deposit
            self.avg_short_price = (
                self.avg_short_price * self.short_volume + order.price * order.volume
            ) / (self.short_volume + order.volume)
            self.short_volume += order.volume
        
        return True, '[Log]: Order executed'
    
    def close_order(self, order: OrderInfo) -> tuple[bool, str]:
        if order.order_direction == 'long':
            if order.volume > self.long_volume:
                return False, '[Error]: Not enough long volume'
            
            self.cash_value += (order.deposit + order.profit)
            self.long_asset_value -= order.deposit
            self.long_volume -= order.volume
            # ------ don't update average_long_price, assume buying all volume at average_long_price ------
            if self.long_volume == 0:
                self.avg_long_price = 0
                
        else:
            if order.volume > self.short_volume:
                return False, '[Error]: Not enough short volume'
            
            self.cash_value += (order.deposit + order.profit)
            self.short_asset_value -= order.deposit
            self.short_volume -= order.volume
            # ------ don't update average_short_price, assume selling all volume at average_short_price ------
            if self.short_volume == 0:
                self.avg_short_price = 0
                
        return True, '[Log]: Order closed'
    