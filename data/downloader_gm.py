import platform
import os
import sys
if platform.system() == 'Windows':
    sys.path.append('C:\\Users\\dng\\Desktop\\work\\QuantPipeline')


from datetime import datetime, timedelta
from typing import Literal
from enum import Enum
import pandas as pd
import gm

from data.downloader import Downloader


class GmTradeType(Enum):
    OPPOSITE_OPEN = 1
    OPPOSITE_CLOSE = 2
    LONG_OPEN = 3
    LONG_CLOSE = 4
    SHORT_OPEN = 5
    SHORT_CLOSE = 6
    LONG_EXCHANGE = 7
    SHORT_EXCHANGE = 8


class GmAdjustType(Enum):
    ADJUST_NONE = 0
    ADJUST_PREV = 1
    ADJUST_POST = 2


class GmQuote(object):
    def __init__(self, bid_p: float, bid_v: float, ask_p: float, ask_v: float):
        self.bid_p = bid_p
        self.bid_v = bid_v
        self.ask_p = ask_p
        self.ask_v = ask_v


class FutureGmDownloader(Downloader):
    """
    主力连续/次主力连续:
        1.由真实合约拼接
        2.日成交量和持仓量都为最大的合约，确定为新的主力合约，每日收盘结算后判断，于下一交易日进行指向切换，日内不会进行主力合约的切换
    月份连续:
        1.由真实合约拼接，由每日的未来合约拼接
        2.00 对应最近月份合约,01 对应其后一个合约,02 对应再后一个合约，依次类推
    指数连续:
        1.由真实合约的累计持仓量加权计算合成
    """
    
    token = 'f3a28e010fcdfbaf076c9f742b633cd4fd25e6f7'
    sec_type1 = 1040
    sec_type2 = 104003
    exchanges_code = ['SHFE', 'DCE', 'CZCE', 'GFEX']
    symbol_fields = ['symbol', 'exchange', 'sec_id', 'sec_name', 'price_tick']
    
    bar_fields = [
        'symbol', 'open', 'low', 'high', 'close', 'volume', 'amount', # basic information
        'position',   # cum position
        'bob', 'eob'  # begin datetime of bar, end datetime of bar
    ]
    bar_real_times = ['15s', '30s', '60s', '300s', '900s', '1800s', '1d']
    bar_history_times = ['60s', '300s', '600s', '1800s', '3600s', '1d', '5d']  # Multiples of 60s
    
    tick_fields = [
        'price', 'last_volume', 'trade_type', 'quotes',  # last trade/moment information
        'cum_volume', 'cum_amount', 'cum_position',  # cum information
        'created_at'  # timestamp
    ]

    def __init__(self, save_dir: str):
        super().__init__(save_dir)
        gm.api.set_token(self.token)

    def download_symbol_info(self, **kwargs) -> pd.DataFrame:
        """download symbol information: symbol, exchange, sec_id, sec_name, price_tick"""
        exchanges = kwargs.get('exchanges', self.exchanges_code)
        symbol_df = gm.api.get_symbol_infos(
            sec_type1=self.sec_type1, sec_type2=self.sec_type2, exchanges=exchanges, df=True
        )
        symbol_df = symbol_df[self.symbol_field]
        return symbol_df

    def download_tick_fun(self, symbol: str, start_time: datetime, end_time: datetime, **kwargs) -> pd.DataFrame:
        return  gm.api.history(
            symbol=symbol, frequency='tick', start_time=start_time, end_time=end_time, 
            fields=','.join(kwargs.get('fields', self.tick_fields)), 
            adjust=kwargs.get('adjust', GmAdjustType.ADJUST_NONE), 
            skip_suspended=True, df=True
        )
    
    def download_bar_fun(self, symbol: str, start_time: datetime, end_time: datetime, freq: str, **kwargs) -> pd.DataFrame:
        return gm.api.history(
            symbol=symbol, frequency=freq, start_time=start_time, end_time=end_time, 
            fields=','.join(kwargs.get('fields', self.bar_fields[1:])), 
            skip_suspended=True, adjust=kwargs.get('adjust', GmAdjustType.ADJUST_PREV), df=True
        )

    def download_ticks_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        res_dicts = gm.api.current(
            symbols, 
            fields=','.join(kwargs.get('fields', self.tick_fields)), 
            include_call_auction=False  #  是否支持集合竞价
        )
        return pd.DataFrame(res_dicts)
    
    def download_bars_real_time(self, symbols: list[str], **kwargs) -> pd.DataFrame:
        res_dicts = gm.api.get_symbols(
            symbols, 
            sec_type1=self.sec_type1,
            sec_type2=self.sec_type2,
            exchanges=kwargs.get('exchanges', self.exchanges_code),
            fields=','.join(kwargs.get('fields', self.bar_fields)), 
            skip_suspended=True,
            trade_date=None,
            df=True
        )
        return pd.DataFrame(res_dicts)


class StockGmDownloader(Downloader):
    token = 'f3a28e010fcdfbaf076c9f742b633cd4fd25e6f7'
    sec_type1 = 1010
    sec_type2 = '101001'
    # exchanges_code = ['SH', 'SZ']
    
    symbol_fields = ['symbol', 'exchange', 'sec_id', 'sec_name', 'price_tick']
    
    bar_fields = [
        'symbol', 'open', 'low', 'high', 'close', 'volume', 'amount', # basic information
        'bob', 'eob'  # begin datetime of bar, end datetime of bar
    ]
    bar_real_times = ['15s', '30s', '60s', '300s', '900s', '1800s', '1d']
    bar_history_times = ['60s', '300s', '600s', '1800s', '3600s', '1d', '5d']  # Multiples of 60s
    
    tick_fields = [
        'price', 'trade_type', 'quotes',  # basic information
        'last_volume',  # last trade/moment information
        'cum_volume', 'cum_amount', 'cum_position',  # cum information
        'created_at'  # timestamp
    ]
    
    def __init__(self, save_dir: str):
        super().__init__(save_dir)
        gm.api.set_token(self.token)


class EtfGmDownloader(Downloader):
    raise NotImplementedError


"""
class DownloaderGm(Downloader):
    def __init__(self, save_dir: str, token: str):
        super().__init__(save_dir)
        set_token(token)

    def download_tick(self, symbol: str, start_time: datetime, end_time: datetime, time_interval: timedelta):
        pass

    def download_bar(
            self, symbol: str, start_time: datetime, end_time: datetime, freq: str, time_interval: timedelta,
            adjust: Literal[ADJUST_PREV, ADJUST_NONE, ADJUST_POST] = ADJUST_NONE
    ):
        df_list = []
        while start_time < end_time:
            df = history(
                symbol=symbol,
                start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                frequency=freq,
                fields=self.bar_field,
                adjust=adjust,
                df=True
            )
            start_time += time_interval
            if df is None or df.empty:
                break
            df_list.append(df)
        return pd.concat(df_list)

    def download_tick(self, symbol: str, start_time: datetime, end_time: datetime, time_interval: timedelta):
        raise NotImplementedError
"""
