import os
import pandas as pd
import numpy as np
from numpy import random
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.tsa.vector_ar.vecm import VECM, VECMResults, coint_johansen, JohansenTestResult
from statsmodels.tsa.vector_ar.vecm import select_coint_rank, CointRankResults

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer

from data.core_utils import H5FileSupporter
from data_cell.core_utils import IndicatorUtils


# 为什么我要去除波动性?
# 数据的问题在于它是不均匀的，体现在：
#   1.不同标的的时间段不一致
#   2.同一标的时间分隔长度不一致，结束时段与下一开始时段的间隔很长
#   3.即便时间间隔长度一致，时间含义依然不均匀


class DataCell(object):
    def __init__(self, name, cache_dir):
        self.name = name
        self.cache_dir = cache_dir
        self.index_path = os.path.join(self.cache_dir, f'{self.name}_index.h5')
        self.norm_dict: Dict[str, pd.DataFrame] = {}
        self.label_dict: Dict[str, pd.DataFrame] = {}
        self.fea_dict: Dict[str, pd.DataFrame] = {}
        self.cache_dir = cache_dir
        self.index_filename = os.path.join(self.cache_dir, f'{self.name}_index.h5')
        self.norm_filename = os.path.join(self.cache_dir, f'{self.name}_norm.h5')
        self.ml_filename = os.path.join(self.cache_dir, f'{self.name}_ml.h5')

    def merge_contract_index(
            self,
            contract_path, symbol_path,
            symbol_filter, contract_filter
    ):
        symbol_dfs = H5FileSupporter.read_h5(symbol_path)
        contract_df: Dict[str, pd.DataFrame] = H5FileSupporter.read_h5(contract_path)
        contract_df: pd.DataFrame = pd.concat(list(contract_df.values()), axis=0, ignore_index=True)
        if contract_filter is not None:
            contract_df = contract_df[contract_df['sec_name'].str.contains('|'.join(contract_filter), na=False)]
        if symbol_filter is not None:
            contract_df = contract_df[contract_df['sec_name'].str.contains('|'.join(symbol_filter), na=False)]
        with pd.HDFStore(self.index_path, mode='w') as store:
            for _, sec_df in contract_df.iterrows():
                df: pd.DataFrame = symbol_dfs[f'/{sec_df.symbol}']
                store.put(sec_df.symbol, df)

    def prepare_norm_data(self, force_update=False):
        if os.path.exists(self.norm_filename) and (not force_update):
            self.norm_dict = H5FileSupporter.read_h5(self.norm_filename)
        else:
            h5data = H5FileSupporter.read_h5(self.index_filename)
            for symbol_key, symbol_df in h5data.items():
                if symbol_df.shape[0] < 1500:
                    continue
                norm_df = self.normal_data(symbol_df)
                norm_df = IndicatorUtils.compute_macd(norm_df)
                norm_df = IndicatorUtils.compute_kdj(norm_df)
                norm_df = IndicatorUtils.compute_ma(norm_df)
                self.norm_dict[symbol_key] = norm_df
            H5FileSupporter.write_h5(self.norm_dict, self.norm_filename)

    @staticmethod
    def prepare_predict_dataset(df: pd.DataFrame):
        df['bob'] = pd.to_datetime(df['bob'])
        df.sort_values(by=['bob'], inplace=True)

        norm_df = pd.DataFrame({
            'datetime': df['bob'],
            'close': df['close'],
            # 'close': (df['high'] + df['low'] + df['close']) / 3.0  # 这里我觉得适用于波动率套利重心
        })

        # ------ 由于close是非稳态的,因此采用rate
        norm_df['rate'] = norm_df['close'] / norm_df['close'].shift(1) - 1.0
        norm_df.iloc[0, norm_df.columns.get_loc('rate')] = 0.0

        norm_df['rate'] = np.maximum(norm_df['rate'], norm_df['rate'].quantile(0.01))
        norm_df['rate'] = np.minimum(norm_df['rate'], norm_df['rate'].quantile(0.99))

        # ------ 由于时间是不均匀的,记录时间的标签
        norm_df['hour'] = norm_df['datetime'].dt.hour
        norm_df['weekday'] = norm_df['datetime'].dt.weekday
        # norm_df['day'] = norm_df['datetime'].dt.day

        norm_df.set_index(keys=['datetime'], inplace=True)
        norm_df.sort_index(axis='rows', ascending=True, inplace=True)

        # ------ 部分均匀化操作
        norm_df['rate_1'] = 0.01 / norm_df['rate'].std() * norm_df['rate']
        norm_df['close_1'] = (norm_df['rate'] + 1.0).cumprod()

        # 去除局部波动性，序列更稳态,应用GARCH去除聚集波动性更好应该更好
        norm_df['rate_2'] = 0.01 / norm_df['rate'].rolling(90).std() * norm_df['rate']
        norm_df['close_2'] = (norm_df['rate_2'] + 1.0).cumprod()

        # ------ 假设rate已经是平稳序列,无波动性聚集的序列
        norm_df['rate_3'] = (norm_df['rate'] - norm_df['rate'].mean()) / norm_df['rate'].std()
        norm_df['rate_4'] = norm_df.groupby(by=['hour'])['rate'].apply(lambda x: (x-x.mean())/x.std())
        norm_df['rate_5'] = norm_df.groupby(by=['weekday'])['rate'].apply(lambda x: (x - x.mean()) / x.std())



        # norm_df.iloc[0, norm_df.columns.get_loc('rate')] = 1.0
        # norm_df['close'] = norm_df['rate'].cumprod()
        # norm_df['log_close'] = np.log(norm_df['close'])
        # norm_df['volume'] = df['volume'] / df.iloc[0]['volume']
        # norm_df['rate'] = norm_df['rate'] - 1.0
        #
        # norm_df = norm_df[(norm_df['hour'] > 0) & (norm_df['hour'] < 23)]  # 不应该删除,只能在训练时去除

        return norm_df

if __name__ == '__main__':
    cell = DataCell(
        name='dc',
        cache_dir='/home/admin123456/Desktop/work_share/FinTech/QuantPipeline/cell_cache'
    )
    # cell.merge_contract_index(
    #     '/home/admin123456/Desktop/work_share/FinTech/QuantPipeline/download_data/gm_contract_info.h5',
    #     '/home/admin123456/Desktop/work_share/FinTech/QuantPipeline/download_data/gm_symbol_3600s.h5',
    #     symbol_filter=[
    #         '苹果', '玻璃', '甲醇', '纯碱', '烧碱', '白糖', 'PTA', '豆一', '豆二', '胶合板', '铁矿石', '焦炭', '焦煤',
    #         '豆粕', '塑料', 'PVC', '豆油', '沥青', '燃油', '热卷', '螺纹钢', '橡胶', '不锈钢', '线材', '郑棉', '红枣',
    #         '甲醇', '菜粕', '菜油', '菜籽', '硅铁'
    #     ],
    #     contract_filter=['连一'],
    # )
    # cell.prepare_norm_data()

    a = pd.DataFrame(np.random.random((100, 3)), columns=['x', 'y', 'z'])
    print(a['x'].mean())
