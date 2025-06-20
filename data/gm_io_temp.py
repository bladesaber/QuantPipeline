import os
import pandas as pd
import numpy as np
from numpy import random
from typing import List, Dict, Union, Literal
from functools import partial

import gm
from gm.api import set_token, history_n, history
from gm.api import ADJUST_PREV, ADJUST_NONE, ADJUST_POST

from utils.multiprocess_utils import apply_async_run_with_res
from data.core_utils import Downloader, H5FileSupporter


class GmIO(object):
    sec_type1 = 1040
    sec_type2 = 104003
    exchanges_code = {
        '上期所': 'SHFE',
        '大商所': 'DCE',
        '郑商所': 'CZCE',
        '广期所': 'GFEX'
    }
    tick_field = ['symbol', 'open', 'high', 'low', 'price', 'trade_type', 'last_volume', 'quotes']  # quotes指买卖十档
    quotes_field = ['bid_p', 'bid_v', 'ask_p', 'ask_v']
    bar_field = ['symbol', 'open', 'low', 'high', 'close', 'volume', 'bob', 'eob']
    freq_field = ['tick', '1d', '60s', '600s', '1800s', '3600s']
    token = 'f3a28e010fcdfbaf076c9f742b633cd4fd25e6f7'

    def __init__(self, save_dir: str, contract_info_file=None, symbol_data_filename=None):
        set_token(self.token)
        self.save_dir = save_dir

        if contract_info_file is None:
            self.contract_info_file = os.path.join(self.save_dir, 'gm_contract_info.h5')
        else:
            self.contract_info_file = contract_info_file

        if symbol_data_filename is None:
            self.symbol_data_filename = 'gm_symbol'
        else:
            self.symbol_data_filename = symbol_data_filename

    def get_symbol_freq_data_file(self, freq):
        return os.path.join(self.save_dir, f"{self.symbol_data_filename}_{freq}.h5")

    def update_contract_info(
            self, filter_str: List[str] = ['主力', '连一'],
            columns=['symbol', 'sec_type1', 'sec_type2', 'exchange', 'sec_id', 'sec_name']
    ):
        dfs_dict = {}
        for exchange, code in self.exchanges_code.items():
            print(f'[INFO]: Updating {exchange} contract info...')
            symbol_df = gm.api.get_symbol_infos(self.sec_type1, self.sec_type2, exchanges=code, df=True)
            symbol_df = symbol_df[symbol_df['sec_name'].str.contains('|'.join(filter_str), na=False)]
            symbol_df = symbol_df[columns]
            dfs_dict[code] = symbol_df
        H5FileSupporter.write_h5(dfs_dict, self.contract_info_file)

    def history_n_fun(self, param: Dict, with_token=False):
        if with_token:
            set_token(self.token)
        return param['key'], history_n(**param['value'])

    def history_fun(self, param: Dict, with_token=False):
        if with_token:
            set_token(self.token)
        return param['key'], history(**param['value'])

    def update_history_n(
            self, steps: int, freq_set: List[Literal['tick', '1d', '60s', '600s', '1800s', '3600s']],
            contract_filter: List[str] = ['连一'], symbol_filter: List[str] = None,
            fields=['symbol', 'open', 'low', 'high', 'close', 'volume', 'bob', 'eob'],
            adjust: Literal[ADJUST_PREV, ADJUST_NONE, ADJUST_POST] = ADJUST_PREV,
            threads=1
    ):
        contract_df_dicts = H5FileSupporter.read_h5(self.contract_info_file)
        for freq in freq_set:
            symbol_dfs = {}
            param_list = []
            for exchange, contract_df in contract_df_dicts.items():
                select_df = contract_df.copy()
                if contract_filter is not None:
                    select_df: pd.DataFrame = select_df[
                        select_df['sec_name'].str.contains('|'.join(contract_filter), na=False)
                    ]
                if symbol_filter is not None:
                    select_df: pd.DataFrame = select_df[
                        select_df['sec_name'].str.contains('|'.join(symbol_filter), na=False)
                    ]
                for _, item in select_df.iterrows():
                    param_list.append({
                        'key': item.symbol,
                        'value': {
                            'symbol': item.symbol,
                            'frequency': freq,
                            'count': steps,
                            'fields': ','.join(fields),
                            'adjust': adjust,
                            'df': True
                        }
                    })
            if len(param_list) > 0:
                if threads == 1:
                    for param_info in param_list:
                        print(f"[INFO]: Updating {param_info['key']} with {steps} count of {freq}...")
                        symbol_key, symbol_df = self.history_n_fun(param_info, with_token=False)
                        symbol_dfs[symbol_key] = symbol_df
                else:
                    res_list = apply_async_run_with_res(
                        pool_num=threads,
                        global_func=partial(self.history_n_fun, with_token=True),
                        params=param_list, save_res=True
                    )
                    for symbol_key, symbol_df in res_list:
                        symbol_dfs[symbol_key] = symbol_df
                H5FileSupporter.write_h5(symbol_dfs, self.get_symbol_freq_data_file(freq))

    def update_history(
            self, freq_set: List[Literal['tick', '1d', '60s', '600s', '1800s', '3600s']],
            start_time='2022-01-01 09:00:00', end_time='2025-01-01 09:00:00',
            contract_filter: List[str] = ['连一'], symbol_filter: List[str] = None,
            fields=['symbol', 'open', 'low', 'high', 'close', 'volume', 'bob', 'eob'],
            adjust: Literal[ADJUST_PREV, ADJUST_NONE, ADJUST_POST] = ADJUST_NONE,
            threads=1
    ):
        contract_df_dicts = H5FileSupporter.read_h5(self.contract_info_file)
        for freq in freq_set:
            symbol_dfs = {}
            param_list = []
            for exchange, contract_df in contract_df_dicts.items():
                select_df = contract_df.copy()
                if contract_filter is not None:
                    select_df: pd.DataFrame = select_df[
                        select_df['sec_name'].str.contains('|'.join(contract_filter), na=False)
                    ]
                if symbol_filter is not None:
                    select_df: pd.DataFrame = select_df[
                        select_df['sec_name'].str.contains('|'.join(symbol_filter), na=False)
                    ]
                for _, item in select_df.iterrows():
                    param_list.append({
                        'key': item.symbol,
                        'value': {
                            'symbol': item.symbol,
                            'frequency': freq,
                            'start_time': start_time,
                            'end_time': end_time,
                            'fields': ','.join(fields),
                            'adjust': adjust,
                            'df': True
                        }
                    })

            if len(param_list) > 0:
                if threads == 1:
                    for param_info in param_list:
                        print(f"[INFO]: Updating {param_info['key']} of {freq}...")
                        symbol_key, symbol_df = self.history_fun(param_info, with_token=False)
                        symbol_dfs[symbol_key] = symbol_df
                else:
                    res_list = apply_async_run_with_res(
                        pool_num=threads,
                        global_func=partial(self.history_fun, with_token=True),
                        params=param_list, save_res=True
                    )
                    for symbol_key, symbol_df in res_list:
                        symbol_dfs[symbol_key] = symbol_df
                H5FileSupporter.write_h5(symbol_dfs, self.get_symbol_freq_data_file(freq))

    def download_tick(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

if __name__ == '__main__':
    downloader = GmIO(save_dir='D:/QuantPipeline/download_data')

    # downloader.update_contract_info(filter_str=[
    #     '主力', '连一', '连二', '连三', '连四', '连五'
    # ])
    # contract_df_dicts = H5FileSupporter.read_h5(downloader.contract_info_file)
    # for exchange, df in contract_df_dicts.items():
    #     print(df)

    # downloader.update_history_n(steps=10000, freq_set=['3600s', '1800s', '600s'], threads=10)
    downloader.update_history(
        freq_set=['1d', '3600s', '1800s', '300s'], threads=3,
        symbol_filter=[
            '苹果', '玻璃', '甲醇', '纯碱', '烧碱', '白糖', 'PTA', '豆一', '豆二', '胶合板', '铁矿石', '焦炭', '焦煤',
            '豆粕', '塑料', 'PVC', '豆油', '沥青', '燃油', '热卷', '螺纹钢', '橡胶', '不锈钢', '线材', '郑棉', '红枣',
            '甲醇', '菜粕', '菜油', '菜籽', '硅铁'
        ],
        contract_filter=['连一', '连二', '连三', '连四', '连五'],
        start_time='2018-01-01 09:00:00', end_time='2025-01-01 09:00:00'
    )
