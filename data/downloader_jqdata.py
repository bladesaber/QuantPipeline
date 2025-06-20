import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from colorama import Fore, Back, Style, init
import jqdatasdk as jq

from data.downloader import Downloader, BarTask, TickTask


init(autoreset=True)


class FutureJqDownloader(Downloader):
    username = '13415586883'
    password = 'Qq2510294705'
    
    exchanges_code = ['CCFX', 'XDCE', 'XSGE', 'XZCE']
    symbol_fields = ['symbol', 'display_name']
    
    bar_fields = [
        'security', 'open', 'low', 'high', 'close', 'volume', 'money', # basic information
        'open_interest',   # cum position
        'date'  # begin datetime of bar, end datetime of bar
    ]
    bar_times = ['0.25m', '0.5m', '1m', '5m', '15m', '30m', '60m' '1d']
    
    tick_fields = [
        'security', 'time', 'current',  # last trade/moment information
        'volume', 'money', 'position',  # cum information
        'a1_v', 'a1_p', 'b1_v', 'b1_p'  # 一档卖量, 一档卖价, 一档买量, 一档买价
    ]

    def __init__(self):
        jq.auth(self.username, self.password)

    def download_symbol_info(self, **kwargs) -> pd.DataFrame:
        symbols_df = jq.get_all_securities(types=['futures'], date=kwargs.get('date', datetime.now().strftime('%Y-%m-%d')))
        symbols_df.index.name = 'symbol'
        symbols_df.reset_index(inplace=True)
        symbols_df = symbols_df[self.symbol_fields]
        return symbols_df
    
    def download_tick_fun(self, symbol: str, start_time: datetime, end_time: datetime, **kwargs) -> pd.DataFrame:
        return jq.get_ticks(
            security=symbol,
            start_dt=Downloader.convert_to_jq_time(start_time), end_dt=Downloader.convert_to_jq_time(end_time),
            count=None,
            df=True
        )
    
    def download_bar_fun(self, symbol: str, start_time: datetime, end_time: datetime, freq: str, **kwargs) -> pd.DataFrame:
        return jq.get_bars(
            security=symbol, 
            start_dt=Downloader.convert_to_jq_time(start_time), end_dt=Downloader.convert_to_jq_time(end_time),
            unit=freq,
            fields=kwargs.get('fields', self.bar_fields[1:]),
            include_now=kwargs.get('include_now', True),
            count=None,
            df=True
        )
    
    def download_ticks_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        raise NotImplementedError('jqdata does not support real-time ticks download, only support online real-time deployment')
    
    def download_bars_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        raise NotImplementedError('jqdata does not support real-time bars download, only support online real-time deployment')
    
    def download_multi_ticks(self, tasks: list[TickTask], pool_num: int = -1, threads: int = -1) -> dict:
        assert pool_num <= 0 and threads <= 0, 'jqdata does not support multi-threaded download'
        super().download_multi_ticks(tasks, pool_num, threads)
    
    def download_multi_bars(self, tasks: list[BarTask], pool_num: int = -1, threads: int = -1) -> dict:
        assert pool_num <= 0 and threads <= 0, 'jqdata does not support multi-threaded download'
        super().download_multi_bars(tasks, pool_num, threads)


class StockJqDownloader(Downloader):
    username = '13415586883'
    password = 'Qq2510294705'
    
    exchanges_code = ['XSHG', 'XSHE', 'BJSE']
    symbol_fields = ['symbol', 'display_name', 'start_date', 'end_date']

    def __init__(self):
        jq.auth(self.username, self.password)

    def download_symbol_info(self, **kwargs) -> pd.DataFrame:
        symbols_df = jq.get_all_securities(types=['stock'], date=kwargs.get('date', datetime.now().strftime('%Y-%m-%d')))
        symbols_df.index.name = 'symbol'
        symbols_df.reset_index(inplace=True)
        symbols_df = symbols_df[self.symbol_fields]
        return symbols_df

    def download_multi_ticks(self, tasks: list[TickTask], pool_num: int = -1, threads: int = -1) -> dict:
        assert pool_num <= 0 and threads <= 0, 'jqdata does not support multi-threaded download'
        super().download_multi_ticks(tasks, pool_num, threads)

    def download_multi_bars(self, tasks: list[BarTask], pool_num: int = -1, threads: int = -1) -> dict:
        assert pool_num <= 0 and threads <= 0, 'jqdata does not support multi-threaded download'
        super().download_multi_bars(tasks, pool_num, threads)
