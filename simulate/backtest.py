import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import time
from colorama import Fore, Back, Style, init

from sim_analysis.backtest_utils import MultiPeriodBars, OrderInfo, AccountStatistic

init(autoreset=True)
        

class FutureOrderBacktest(ABC):
    def __init__(
        self, 
        multi_period_bars: MultiPeriodBars, 
        backward_window: int,
        asset_inital_value: float, deposit_ratio: float, 
        context_cost: float, slippage: float = -1,
    ):
        np.random.seed(time.time())
        self.multi_period_bars = multi_period_bars
        self.backward_window = backward_window
        self.deposit_ratio = deposit_ratio
        self.context_cost = context_cost
        self.slippage = slippage
        self.order_id = 0
        
        self.existing_orders: dict[int, OrderInfo] = {}
        self.finished_orders: dict[int, OrderInfo] = {}
        self.account_stat = AccountStatistic(asset_inital_value)

    @abstractmethod
    def _preprocess(self, multi_period_bars: MultiPeriodBars):
        self.existing_orders.clear()
        self.finished_orders.clear()
        self.account_stat.reset()
        self.order_id = 0

    def run(self, order_events: pd.DataFrame = None):
        self._preprocess(self.multi_period_bars)
        backward_window = self.backward_window if order_events is None else 0
        
        for iloc in range(len(self.multi_period_bars)):
            if iloc < backward_window:
                continue
            
            self.multi_period_bars.update_scence(iloc, backward_window)
            timestamp = self.multi_period_bars.get_timestamp(iloc)
            
            if order_events is None:
                features = self._get_features(self.multi_period_bars.scence_bars)
                order_infos = self._make_decision(features, self.account_stat.to_dict())
            else:
                order_infos = []
                for event in order_events.loc[timestamp]:
                    order_info = OrderInfo(
                        order_id=-1,
                        date_time=timestamp,
                        order_direction=event['order_direction'],
                        order_style='none',
                        price=event['price'],
                        volume=event['volume'],
                        stop_loss_price=event['stop_loss_price'],
                        take_profit_price=event['take_profit_price'],
                    )
                    order_infos.append(order_info)
                    
            self._execute_decision(timestamp, order_infos)
            self._execute_engine(self.multi_period_bars.get_main_bar(iloc))

    @abstractmethod
    def _get_features(self, period_bars: dict[str, pd.DataFrame]) -> dict:
        raise NotImplementedError('Please implement _get_features method')

    @abstractmethod
    def _make_decision(self, features: dict, account_stat_dict: dict) -> list[OrderInfo]:
        raise NotImplementedError('Please implement _make_decision method')

    def _execute_decision(self, timestamp: pd.Timestamp, order_infos: list[OrderInfo]):
        for order in order_infos:
            order.order_id = self.order_id
            self.order_id += 1

            order.deposit = order.price * self.deposit_ratio * order.volume
            if self.slippage > 0:
                order.slippage_cost = np.random.uniform(0.0, 1.0) * self.slippage * order.price * order.volume
            else:
                order.slippage_cost = 0.0
            order.context_cost = self.context_cost * order.volume
            
            is_success, msg = self.account_stat.open_order(order)
            if not is_success:
                order.status = OrderInfo.execute_status[2]
                order.msg = msg
                order.close_date_time = timestamp
                self.finished_orders[order.order_id] = order
                print(Fore.RED + f'Order {order.order_id}: {msg}')
            else:
                order.status = OrderInfo.execute_status[0]
                self.existing_orders[order.order_id] = order
        
    def _execute_engine(self, bar: pd.Series):
        for order_id in list(self.existing_orders.keys()):
            order = self.existing_orders.pop(order_id)
            if order.order_direction == 'long':                    
                order.is_triggered_sl = bar.price < order.stop_loss_price
                order.is_triggered_tp = bar.price > order.take_profit_price
                if (not order.is_triggered_sl) and (not order.is_triggered_tp):
                    continue
                
                if order.is_triggered_sl and order.is_triggered_tp:
                    # ------ seems uncertain order as loss order
                    order.status = OrderInfo.execute_status[2]
                    order.profit = (order.stop_loss_price - order.price) * order.volume
                    order.msg = '[Error]: Stop loss and take profit are both triggered'
                    print(Fore.RED + f'Order {order_id}: {order.msg}')
                else:
                    order.status = OrderInfo.execute_status[1]
                    if order.is_triggered_sl:
                        order.profit = (order.stop_loss_price - order.price) * order.volume
                    else:
                        order.profit = (order.take_profit_price - order.price) * order.volume
                    
            elif order.order_direction == 'short':
                order.is_triggered_sl = bar.price > order.stop_loss_price
                order.is_triggered_tp = bar.price < order.take_profit_price
                if (not order.is_triggered_sl) and (not order.is_triggered_tp):
                    continue
                
                if order.is_triggered_sl and order.is_triggered_tp:
                    order.status = OrderInfo.execute_status[2]
                    order.profit = (order.price - order.stop_loss_price) * order.volume
                    order.msg = '[Error]: Stop loss and take profit are both triggered'
                    print(Fore.RED + f'Order {order_id} is triggered stop loss and take profit')
                else:
                    order.status = OrderInfo.execute_status[1]
                    if order.is_triggered_sl:
                        order.profit = (order.price - order.stop_loss_price) * order.volume
                    else:
                        order.profit = (order.price - order.take_profit_price) * order.volume
            
            is_success, msg = self.account_stat.close_order(order)
            assert is_success, f'Order {order_id} close failed: {msg}'

            order.close_date_time = bar.date_time
            self.finished_orders[order_id] = order

    def export_result(self, event_file_path: str=None, info_file_path: str=None):
        all_orders = list(self.existing_orders.values()) + list(self.finished_orders.values())
        
        if info_file_path is not None:
            info_items = []
            for order in all_orders:
                info_items.append({
                    'order_id': order.order_id, 'date_time': order.date_time, 'close_date_time': order.close_date_time, 
                    'order_direction': order.order_direction, 'price': order.price, 'volume': order.volume, 
                    'stop_loss_price': order.stop_loss_price, 'take_profit_price': order.take_profit_price, 'profit': order.profit, 
                    'slippage_cost': order.slippage_cost, 'context_cost': order.context_cost, 'deposit': order.deposit, 
                    'status': order.status, 'msg': order.msg
                })
            info_df = pd.DataFrame(info_items)
            del info_items
            info_df.to_csv(info_file_path, index=False)

        if event_file_path is not None:
            event_items = []
            for order in all_orders:
                event_items.append({
                    'date_time': order.date_time, 
                    'order_direction': order.order_direction,
                    'volume': order.volume, 
                    'price': order.price, 
                    'order_cash': -(order.deposit + order.slippage_cost + order.context_cost), 
                    'order_trade_cost': -order.slippage_cost - order.context_cost,
                    'order_asset': order.deposit,
                    'profit': order.profit,
                    'order_id': order.order_id, 
                    'order_style': 'open'
                })

                if order.status == OrderInfo.execute_status[1]:
                    close_item = {
                        'date_time': order.close_date_time, 
                        'order_direction': order.order_direction,
                        'volume': -order.volume, 
                        'price': -1, 
                        'order_cash': order.profit + order.deposit, 
                        'order_trade_cost': 0.0,
                        'order_asset': -order.deposit,
                        'profit': order.profit,
                        'order_id': order.order_id, 
                        'order_style': 'close'
                    }
                    if order.is_triggered_tp:
                        close_item['price'] = order.take_profit_price
                    elif order.is_triggered_sl:
                        close_item['price'] = order.stop_loss_price
                    event_items.append(close_item)

            event_df = pd.DataFrame(event_items)
            event_df.to_csv(event_file_path, index=False)


class FutureSeqBacktest(ABC):
    def __init__(
        self, 
        multi_period_bars: MultiPeriodBars, 
        backward_window: int,
        asset_inital_value: float, deposit_ratio: float, 
        context_cost: float, slippage: float = -1,
    ):
        self.multi_period_bars = multi_period_bars
        self.backward_window = backward_window
        self.deposit_ratio = deposit_ratio
        self.context_cost = context_cost
        self.slippage = slippage
        np.random.seed(time.time())
        
        self.order_id = 0
        self.existing_orders: dict[int, OrderInfo] = {}
        self.finished_orders: dict[int, OrderInfo] = {}
        self.account_stat = AccountStatistic(asset_inital_value)

    @abstractmethod
    def _preprocess(self, multi_period_bars: MultiPeriodBars):
        self.account_stat.reset()
        self.existing_orders.clear()
        self.finished_orders.clear()
        self.order_id = 0
    
    def run(self, order_events: pd.DataFrame = None):
        self._preprocess(self.multi_period_bars)
        backward_window = self.backward_window if order_events is None else 0
        
        for iloc in range(len(self.multi_period_bars)):
            if iloc < backward_window:
                continue
            
            self.multi_period_bars.update_scence(iloc, backward_window)
            timestamp = self.multi_period_bars.get_timestamp(iloc)
            
            if order_events is None:
                features = self._get_features(self.multi_period_bars.scence_bars)
                order_infos = self._make_decision(features, self.account_stat.to_dict())
            else:
                order_infos = []
                for event in order_events.loc[timestamp]:
                    order_info = OrderInfo(
                        order_id=-1,
                        date_time=timestamp,
                        order_direction=event['order_direction'],
                        order_style=event['order_style'],
                        price=event['price'],
                        volume=event['volume'],
                        stop_loss_price=None,
                        take_profit_price=None,
                    )
                    order_infos.append(order_info)
            
            self._execute_decision(timestamp, order_infos)
            self._execute_engine(self.multi_period_bars.get_main_bar(iloc))

    @abstractmethod
    def _get_features(self, period_bars: dict[str, pd.DataFrame]) -> dict:
        raise NotImplementedError('Please implement _get_features method')

    @abstractmethod
    def _make_decision(self, features: dict, account_stat_dict: dict) -> list[OrderInfo]:
        raise NotImplementedError('Please implement _make_decision method')
    
    def _execute_decision(self, timestamp: pd.Timestamp, order_infos: list[OrderInfo]):
        for order in order_infos:
            order.order_id = self.order_id
            self.order_id += 1
            
            # if it's close order, just add to existing_orders for next engine
            if order.order_style == 'close':
                self.existing_orders[order.order_id] = order
                continue

            order.deposit = order.price * self.deposit_ratio * order.volume
            if self.slippage > 0:
                order.slippage_cost = np.random.uniform(0.0, 1.0) * self.slippage * order.price * order.volume
            else:
                order.slippage_cost = 0.0
            order.context_cost = self.context_cost * order.volume
            
            is_success, msg = self.account_stat.open_order(order)
            if not is_success:
                order.status = OrderInfo.execute_status[2]
                order.msg = msg
                print(Fore.RED + f'Order {order.order_id}: {msg}')
            else:
                order.status = OrderInfo.execute_status[1]
            order.close_date_time = timestamp
            self.finished_orders[order.order_id] = order
    
    def _execute_engine(self, bar: pd.Series):
        for order_id in list(self.existing_orders.keys()):
            order = self.existing_orders.pop(order_id)
            
            if order.order_direction == 'long':
                order.profit = (bar.price - self.account_stat.avg_long_price) * order.volume
            else:
                order.profit = (self.account_stat.avg_short_price - bar.price) * order.volume
            
            is_success, msg = self.account_stat.close_order(order)
            if not is_success:
                order.status = OrderInfo.execute_status[2]
                order.msg = msg
                order.profit = 0.0
                print(Fore.RED + f'Order {order.order_id}: {msg}')
            else:
                order.status = OrderInfo.execute_status[1]
            order.close_date_time = bar.date_time
            self.finished_orders[order_id] = order

    def export_result(self, event_file_path: str=None, info_file_path: str=None):
        if info_file_path is not None:
            info_df = pd.DataFrame(columns=[
                'date_time', 'order_direction', 'price', 'volume', 'profit',
                'slippage_cost', 'context_cost', 'deposit', 
                'status', 'msg'
            ])
            info_items = []
            for order_id, order in self.existing_orders.items():
                info_items.append({
                    'order_id': order.order_id, 'date_time': order.date_time, 'order_direction': order.order_direction, 
                    'price': order.price, 'volume': order.volume, 'profit': order.profit,
                    'slippage_cost': order.slippage_cost, 'context_cost': order.context_cost, 'deposit': order.deposit, 
                    'status': order.status, 'msg': order.msg
                })
            info_df.to_csv(info_file_path, index=False)
        
        if event_file_path is not None:
            event_items = []
            for order_id, order in self.existing_orders.items():
                if order.status != OrderInfo.execute_status[1]:
                    continue
                
                event_item = {
                    'date_time': order.date_time, 
                    'order_direction': order.order_direction,
                    'volume': order.volume, 
                    'price': order.price, 
                    'profit': order.profit,
                    'order_trade_cost': -order.slippage_cost - order.context_cost,
                    'order_id': order.order_id,
                    'order_style': order.order_style
                }
                if order.order_style == 'open':
                    event_item['order_cash'] = -(order.deposit + order.slippage_cost + order.context_cost)
                    event_item['order_asset'] = order.deposit
                elif order.order_style == 'close':
                    event_item['order_cash'] = order.profit + order.deposit
                    event_item['order_asset'] = -order.deposit
                event_items.append(event_item)

            event_df = pd.DataFrame(event_items)
            event_df.to_csv(event_file_path, index=False)
