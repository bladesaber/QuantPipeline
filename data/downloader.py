import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
import os
from abc import ABC, abstractmethod

from utils.parallel_utils import ParallelProcess, ParallelThread
from utils.save_load import SaveLoad


class BarTask(object):
    def __init__(self, symbol: str, start_time: datetime, end_time: datetime, freq: str, time_interval: timedelta, **kwargs):
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.time_interval = time_interval
        self.kwargs: dict = kwargs
        self.data: pd.DataFrame = None


class TickTask(object):
    def __init__(self, symbol: str, start_time: datetime, end_time: datetime, time_interval: timedelta, **kwargs):
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.time_interval = time_interval
        self.kwargs: dict = kwargs
        self.data: pd.DataFrame = None


class Downloader(ABC):
    bar_field = ['symbol', 'datetime', 'open', 'low', 'high', 'close', 'volume']
    tick_field = ['symbol', 'datetime', 'price', 'trade_type', 'quotes']

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def convert_str_to_datetime(str_time: str) -> datetime:
        return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def convert_datetime_to_str(datetime_time: datetime) -> str:
        return datetime_time.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def create_datetime(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
        return datetime(year, month, day, hour, minute, second)

    @staticmethod
    def create_time_interval(days: int, hours: int, minutes: int, seconds: int) -> timedelta:
        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    @abstractmethod
    def download_symbol_info(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("download_symbol_info is not implemented")

    @abstractmethod
    def download_ticks_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        """cross section ticks real time"""
        raise NotImplementedError("download_ticks_real_time is not implemented")
    
    @abstractmethod
    def download_bars_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        """cross section bars real time"""
        raise NotImplementedError("download_bars_real_time is not implemented")

    @abstractmethod
    def download_tick_fun(self, symbol: str, start_time: datetime, end_time: datetime, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("download_history_ticks is not implemented")
    
    @abstractmethod
    def download_bar_fun(self, symbol: str, start_time: datetime, end_time: datetime, freq: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("download_history_bars is not implemented")

    def download_tick(self, symbol: str, start_time: datetime, end_time: datetime, time_interval: timedelta, **kwargs) -> dict:
        data_list: list[pd.DataFrame] = []
        current_time = start_time
        while current_time < end_time:
            end_time = min(current_time + time_interval, end_time)
            data = self.download_tick_fun(symbol=symbol, start_time=current_time, end_time=end_time, **kwargs)
            if data is not None and data.shape[0] > 0:
                data_list.append(data)
            current_time = end_time
        return {'symbol': symbol, 'data': pd.concat(data_list, axis=0, ignore_index=True)}

    def download_bar(self, symbol: str, start_time: datetime, end_time: datetime, freq: str, time_interval: timedelta, **kwargs) -> dict:
        data_list: list[pd.DataFrame] = []
        current_time = start_time
        while current_time < end_time:
            end_time = min(current_time + time_interval, end_time)
            data = self.download_bar_fun(symbol=symbol, start_time=current_time, end_time=end_time, freq=freq, **kwargs)
            if data is not None and data.shape[0] > 0:
                data_list.append(data)
            current_time = end_time
        return {'symbol': symbol, 'data': pd.concat(data_list, axis=0, ignore_index=True)}

    def download_multi_bars(self, tasks: list[BarTask], pool_num: int = -1, threads: int = -1) -> dict:
        if pool_num <= 0 and threads <= 0:
            for task in tasks:
                res = self.download_bar(
                    symbol=task.symbol, start_time=task.start_time, end_time=task.end_time,
                    freq=task.freq, time_interval=task.time_interval, **task.kwargs
                )
                task.data = res['data']
        elif pool_num > 0:
            runner = ParallelProcess(pool_num, self.download_bar)
            task_dict, params = {}, []
            for task in tasks:
                task_dict[task.symbol] = task
                params.append({
                    'symbol': task.symbol,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'freq': task.freq,
                    'time_interval': task.time_interval,
                    **task.kwargs
                })
            res_list = runner.start(params)
            for res in res_list:
                task_dict[res['symbol']].data = res['data']
        else:
            runner = ParallelThread(threads, self.download_bar)
            task_dict, params = {}, []
            for task in tasks:
                task_dict[task.symbol] = task
                params.append({
                    'symbol': task.symbol,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'time_interval': task.time_interval,
                    **task.kwargs
                })
            res_list = runner.start(params)
            for res in res_list:
                task_dict[res['symbol']].data = res['data']

    def download_multi_ticks(self, tasks: list[TickTask], pool_num: int = -1, threads: int = -1) -> dict:
        if pool_num <= 0 and threads <= 0:
            for task in tasks:
                res = self.download_tick(symbol=task.symbol, start_time=task.start_time, end_time=task.end_time, time_interval=task.time_interval, **task.kwargs)
                task.data = res['data']
        elif pool_num > 0:
            runner = ParallelProcess(pool_num, self.download_tick)
            task_dict, params = {}, []
            for task in tasks:
                task_dict[task.symbol] = task
                params.append({
                    'symbol': task.symbol,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'time_interval': task.time_interval,
                    **task.kwargs
                })
            res_list = runner.start(params)
            for res in res_list:
                task_dict[res['symbol']].data = res['data']
        else:
            runner = ParallelThread(threads, self.download_tick)
            task_dict, params = {}, []
            for task in tasks:
                task_dict[task.symbol] = task
                params.append({
                    'symbol': task.symbol,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'time_interval': task.time_interval,
                    **task.kwargs
                })
            res_list = runner.start(params)
            for res in res_list:
                task_dict[res['symbol']].data = res['data']


if __name__ == "__main__":
    pass

    
