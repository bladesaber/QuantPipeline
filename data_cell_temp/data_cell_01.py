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
from data.core_utils import PreprocessLoader


# todo:
#  1.以时间划分模型，更短时间数据,state space model?
#  2.label怎么划分或对齐
#  4.因子有哪几种呢：常人关注的东西? 时间序列? 基本面因子? 其他标的等价关联性? 其他标的上下游关联性?
#  5.自动因子挖掘
#  7.斜率与卡尔曼滤波，检测异常
#  9.当你看一段信息，你会关注什么呢?
#  10.会不会无法预测方向但能评估标的间相对优势或评估方向间的相对优势,stace space model
#  11.关于递减特征（类似强化学习reward）
#  12.有没聚类需求
#  13.异常模型：z-score(分布异常)，卡尔曼滤波(序列异常)，异常信息有没指导价值呢
#  14.要用PCA
#  15.波动性套利?预测正波动幅度，负波动幅度来套利?波动性预测能作为仿真核心吗?
#  16.残差可能有信息
#  17.如何处理缺失数据是个问题?不同品种的时间分段也是个问题?日间断与连续间断也是个问题?
#  18.评估是否应该改为：如果一个模型在多个数据独立训练有效就为有效

class DataCell(object):
    def __init__(self, name, cache_dir):
        self.name = name
        self.norm_dict: Dict[str, pd.DataFrame] = {}
        self.label_dict: Dict[str, pd.DataFrame] = {}
        self.fea_dict: Dict[str, pd.DataFrame] = {}
        self.cache_dir = cache_dir
        self.index_filename = os.path.join(self.cache_dir, f'{self.name}_index.h5')
        self.norm_filename = os.path.join(self.cache_dir, f'{self.name}_norm.h5')
        self.ml_filename = os.path.join(self.cache_dir, f'{self.name}_ml.h5')

    def merge_contract_index(
            self, contract_info_file, symbol_data_file, symbol_filter, contract_filter
    ):
        contract_df: Dict[str, pd.DataFrame] = H5FileSupporter.read_h5(contract_info_file)
        contract_df: pd.DataFrame = pd.concat(list(contract_df.values()), axis=0, ignore_index=True)
        if contract_filter is not None:
            contract_df = contract_df[contract_df['sec_name'].str.contains('|'.join(contract_filter), na=False)]
        if symbol_filter is not None:
            contract_df = contract_df[contract_df['sec_name'].str.contains('|'.join(symbol_filter), na=False)]

        symbol_dfs = H5FileSupporter.read_h5(symbol_data_file)

        index_dfs = {}
        for _, sec_df in contract_df.iterrows():
            index_dfs[f'{sec_df.symbol}'] = symbol_dfs[f'/{sec_df.symbol}']
        H5FileSupporter.write_dfs(index_dfs, self.index_filename)

    # ------ Normal Data:
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
            H5FileSupporter.write_dfs(self.norm_dict, self.norm_filename)

    @staticmethod
    def normal_data(df: pd.DataFrame):
        df['bob'] = pd.to_datetime(df['bob'])
        df['eob'] = pd.to_datetime(df['eob'])
        df.sort_values(by=['bob'], inplace=True)
        # df['mid_c'] = (df['high'] + df['low'] + df['close']) / 3.0

        norm_df = pd.DataFrame({'datetime': df['bob']})

        df['rate'] = df['close'] / df['close'].shift(1)
        df.iloc[0, df.columns.get_loc('rate')] = 1.0

        # ------ 处理极端值
        df['rate'] = np.minimum(np.maximum(df['rate'], df['rate'].quantile(0.01)), df['rate'].quantile(0.99))

        # ------ 使所有序列波动性相同：全局scale与序列局部scale?
        std_scale = 0.01 / np.std(df['rate'] - 1.0)
        norm_df['rate'] = (df['rate'] - 1.0) * std_scale + 1.0
        # abs_scale = np.mean(np.abs(df['rate'].dropna().values - 1.0))
        # norm_df['rate'] = (df['rate'] - 1.0) / abs_scale * 0.001 + 1.0
        # norm_df['rate'] = (df['rate']-1.0) * (0.01 / (df['rate']-1.0).rolling(120).std()) + 1.0

        norm_df.iloc[0, norm_df.columns.get_loc('rate')] = 1.0
        norm_df['close'] = norm_df['rate'].cumprod()
        norm_df['log_close'] = np.log(norm_df['close'])
        norm_df['volume'] = df['volume'] / df.iloc[0]['volume']
        norm_df['rate'] = norm_df['rate'] - 1.0

        norm_df['hour'] = PreprocessLoader.convert_hour(norm_df, key='datetime')
        norm_df['weekday'] = PreprocessLoader.convert_week_day(norm_df, key='datetime')
        # norm_df = norm_df[(norm_df['hour'] > 0) & (norm_df['hour'] < 23)]  # 不应该删除,只能在训练时去除

        norm_df.set_index(keys=['datetime'], inplace=True)
        norm_df.sort_index(axis='rows', ascending=True, inplace=True)

        return norm_df

    # ------ Investigation:
    def investigate_variance_by_length(self):
        """斜率越大说明单调性越好"""
        for symbol, df in self.norm_dict.items():
            var_record = []
            for length in range(1, 200, 2):
                rate: pd.Series = df['close'] / df['close'].shift(length)
                rate = (rate.dropna() - 1.0).values
                rate_std = np.std(rate)
                # rate_std = np.mean(np.abs(rate))
                var_record.append(rate_std)
            var_record = np.array(var_record)
            plt.plot(var_record, '*', label=f"{symbol}")
        # plt.legend()
        plt.show()

    def investigate_statistic_by_hour(self):
        """0时与23时非常不活跃"""
        dfs = []
        for symbol, df in self.norm_dict.items():
            dfs.append(df)
        dfs: pd.DataFrame = pd.concat(dfs, ignore_index=True, axis=0)

        plt.figure()
        for (hour,), sub_df in dfs.groupby(by=['hour']):
            print(f"Hour:{hour} Volume:{np.mean(sub_df['volume'])}")
            plt.scatter(np.ones(sub_df.shape[0]) * hour, sub_df['volume'].values, s=2)
        plt.xlabel('hour')
        plt.ylabel('volume')

        plt.figure()
        for (hour,), sub_df in dfs.groupby(by=['hour']):
            print(f"Hour:{hour} rate:{np.mean(np.abs(sub_df['rate']))}")
            plt.scatter(np.ones(sub_df.shape[0]) * hour, sub_df['rate'].values, s=2)
        plt.xlabel('hour')
        plt.ylabel('rate')

        plt.show()

    def investigate_statistic_by_weekday(self):
        """weekday为5最不活跃"""
        dfs = []
        for symbol, df in self.norm_dict.items():
            dfs.append(df)
        dfs: pd.DataFrame = pd.concat(dfs, ignore_index=True, axis=0)

        plt.figure()
        for (weekday,), sub_df in dfs.groupby(by=['weekday']):
            print(f"Weekday:{weekday} Volume:{np.mean(sub_df['volume'])}")
            plt.scatter(np.ones(sub_df.shape[0]) * weekday, sub_df['volume'].values, s=2)
        plt.xlabel('weekday')
        plt.ylabel('volume')

        plt.figure()
        for (weekday,), sub_df in dfs.groupby(by=['weekday']):
            print(f"Weekday:{weekday} rate:{np.mean(np.abs(sub_df['rate']))}")
            plt.scatter(np.ones(sub_df.shape[0]) * weekday, sub_df['rate'].values, s=2)
        plt.xlabel('weekday')
        plt.ylabel('rate')

        plt.show()

    def investigate_acorr(self):
        """这里看出预测波动性比预测方向更容易，但似乎有微弱的趋势，说明时间序列的直接数据应该有一定效果，但需要PCA"""
        xs, ys = [], []
        for symbol, df in self.norm_dict.items():
            rate: np.ndarray = df['rate'].values
            rate = (rate - np.mean(rate)) / np.std(rate)
            xs.append(rate[:-1])
            ys.append(rate[1:])
        xs = np.concat(xs, axis=0)
        ys = np.concat(ys, axis=0)
        plt.xlabel('ref(x,1)')
        plt.ylabel('x')
        plt.scatter(xs, ys, s=2)
        plt.show()

    def investigate_cross_section_corr(self):
        """这里看出动量与反转效应，越小的rate意味着未来越小的rate，极端的rate则意味着反转，剩下的rate则没明显信息"""
        dfs = []
        for symbol, df in tqdm(self.norm_dict.items()):
            df['symbol'] = symbol
            for step in range(1, 20, 4):
                df[f'pre_rate_{step}'] = df['rate'].rolling(step).sum()
                df[f'fut_rate_{step}'] = df['rate'].shift(-step).rolling(step).sum()
            dfs.append(df.reset_index().dropna())
        dfs: pd.DataFrame = pd.concat(dfs, ignore_index=True, axis=0)

        # for (datetime,), sub_df in tqdm(dfs.groupby(by=['datetime'])):
        #     for step in range(1, 20, 4):
        #         dfs.loc[sub_df.index, f'pre{step}_rank'] = np.argsort(sub_df[f'pre_rate_{step}'])
        #         dfs.loc[sub_df.index, f'fut{step}_rank'] = np.argsort(sub_df[f'fut_rate_{step}'])
        for step in tqdm(range(1, 20, 4)):
            dfs[f'pre{step}_rank'] = dfs.groupby(by=['datetime'])[f'pre_rate_{step}'].rank(method='dense')
            dfs[f'fut{step}_rank'] = dfs.groupby(by=['datetime'])[f'fut_rate_{step}'].rank(method='dense')

        for step in range(1, 20, 4):
            count_df = dfs[[f'pre{step}_rank', f'fut{step}_rank']].value_counts().reset_index()
            heatmap: pd.DataFrame = count_df.pivot(
                index=f'fut{step}_rank', columns=f'pre{step}_rank', values='count'
            ).fillna(0)
            heatmap = heatmap.apply(lambda x: x / np.sum(x), axis=0)
            heatmap = heatmap.clip(lower=0, upper=np.quantile(heatmap, 0.99))

            plt.figure()
            sns.heatmap(heatmap, annot=False, cmap='coolwarm', cbar=True)
            plt.title(f'{step}_heatmap')
        plt.show()

    def investigate_hour_over_hour(self):
        res_dfs = []
        for symbol, df in self.norm_dict.items():
            for (hour,), sub_df in df.groupby(by=['hour']):
                res_df = pd.DataFrame(index=sub_df.index)
                res_df['symbol'] = symbol
                res_df['hour'] = hour

                res_df['rate'] = sub_df['rate']
                res_df['rate_rank'] = KBinsDiscretizer(
                    n_bins=10, encode='ordinal', strategy='quantile'
                ).fit_transform(res_df[['rate']])

                res_df['pre_rate'] = res_df['rate']
                res_df['fut_rate'] = res_df['rate'].shift(-1)
                res_df['pre_rate_rank'] = res_df['rate_rank']
                res_df['fut_rate_rank'] = res_df['rate_rank'].shift(-1)

                res_df.dropna(axis=0, inplace=True)
                res_dfs.append(res_df)
        res_dfs = pd.concat(res_dfs, axis=0, ignore_index=True)

        """看起来有波动性聚集效应，应该由于日间波动性聚集效应导致的"""
        for (hour,), sub_df in res_dfs.groupby(by=['hour']):
            plt.figure()
            plt.scatter(sub_df['pre_rate'], sub_df['fut_rate'], s=2)
            plt.title(f'{hour} hour')
        plt.show()

        """看起来有反转效应"""
        for (hour,), sub_df in res_dfs.groupby(by=['hour']):
            count_df = sub_df[['pre_rate_rank', 'fut_rate_rank']].value_counts().reset_index()
            heatmap: pd.DataFrame = count_df.pivot(
                index='fut_rate_rank', columns='pre_rate_rank', values='count'
            ).fillna(0)
            heatmap = heatmap.apply(lambda x: x / np.sum(x), axis=0)
            heatmap = heatmap.clip(lower=0, upper=np.quantile(heatmap, 0.99))

            plt.figure()
            sns.heatmap(heatmap, annot=False, cmap='coolwarm', cbar=True)
            plt.title(f'{hour}_heatmap')
        plt.show()

    def investigate_objects(self):
        dfs = pd.DataFrame()
        for symbol, df in self.norm_dict.items():
            dfs[f'{symbol}_log_c'] = df['log_close']

        # rank_res: CointRankResults = select_coint_rank(dfs, det_order=0, k_ar_diff=5, signif=0.05)
        # res = VECM(endog=dfs, exog=None, k_ar_diff=5, coint_rank=rank_res.rank, deterministic='co').fit()
        # beta = res.beta.T
        # beta / np.max(beta, axis=1, keepdims=True)
        #
        # heatmap = pd.DataFrame(beta, columns=dfs.columns)
        # plt.figure()
        # sns.heatmap(heatmap, annot=True, cmap='coolwarm', cbar=True)
        # plt.title('Johansen Result')
        # plt.show()

    # ------ Machine Learning Dataset
    def prepare_label(self, beta=0.9):
        for symbol, df in tqdm(self.norm_dict.items()):
            label_df = pd.DataFrame(index=df.index)
            for step in [1, 5, 10]:
                label_df[f'rd_{step}'] = df['rate'].shift(-step).rolling(step).apply(
                    lambda x: np.power(beta, np.arange(0, x.shape[0], 1)).dot(x), raw=True
                )

            for step in [5, 10]:
                label_df[f'rd_{step}_dire'] = np.sign(label_df[f'rd_{step}'])
                label_df[f'rd_{step}_scale'] = np.abs(label_df[f'rd_{step}'])

            for step in [5, 10]:
                label_df[f'rd_{step}_up'] = df['close'].shift(-step).rolling(step).max() / df['close'] - 1.0
                label_df[f'rd_{step}_down'] = df['close'].shift(-step).rolling(step).min() / df['close'] - 1.0

            for step in [5, 10, 20]:
                label_df[f'rd_{step}_slope'] = df['close'].shift(-step).rolling(step).apply(
                    lambda ys: np.polyfit(np.arange(0, step, 1), ys, deg=1)[0]
                )

            self.label_dict[symbol] = label_df

    def prepare_features(self):
        for symbol, df in self.norm_dict.items():
            fea_df = pd.DataFrame(index=df.index)

            # ------ indicator factor
            fea_df['f_dea'] = df['dea']
            fea_df['f_macd'] = df['macd']
            fea_df['f_k'] = df['k']

            # ------ normal factor
            for step in [1, 3, 5, 10, 15, 20]:
                fea_df[f'reward_{step}'] = df['rate'].rolling(step).apply(
                    lambda x: np.power(0.9, np.arange(0, x.shape[0], 1)[::-1]).dot(x), raw=True
                )

            for step in [5, 10, 20, 40]:
                fea_df[f'f_close_lag_{step}'] = df['close'] / df['close'].shift(step) - 1.0

            for ma1, ma2 in [
                (5, 10), (5, 20), (10, 20), (10, 40), (20, 40)
            ]:
                fea_df[f'f_ma{ma1}_cm_ma{ma2}'] = df[f'ma_({ma1}'] / df[f'ma_{ma2}'] - 1.0

            for step in range(1, 20, 1):
                fea_df[f'f_rate_{step}'] = df['rate'].shift(step)

            for step in [5, 10, 20]:
                fea_df[f'f_slope_{step}'] = df['close'].rolling(step).apply(
                    lambda ys: np.polyfit(np.arange(0, step, 1), ys, deg=1)[0]
                )
                fea_df[f'f_intercept_{step}'] = df['close'].rolling(step).apply(
                    lambda ys: np.polyfit(np.arange(0, step, 1), ys, deg=1)[1]
                )

            self.fea_dict[symbol] = fea_df

            # fea_df.reset_index(inplace=True)
            # fea_df['symbol'] = symbol.strip().replace('/', '')
            # fea_dfs.append(fea_df)

        # fea_list = list(set(fea_list))
        # fea_dfs = pd.concat(fea_dfs, ignore_index=True)
        # fea_dfs.dropna(axis=0, inplace=True)
        #     'df': ml_dfs,
        #     'f_list': pd.Series([
        #         'f_dea', 'f_ma5_lag1', 'f_ma10_lag1', 'f_ma20_lag1', 'f_ma40_lag1',
        #         'f_ma5_cm_ma20', 'f_ma5_cm_ma10', 'f_ma10_cm_ma20', 'f_ma20_cm_ma40', 'f_ma40_cm_ma60',
        #     ]),
        #     'l_list': pd.Series([
        #         'l_c_lag1', 'l_c_lag5', 'l_c_lag10', 'l_c_lag20',
        #         # 'l_c_low5', 'l_c_low10', 'l_c_low20'
        #     ])
        # }
        #
        # # todo 目标可以是相关性一致
        # H5FileSupporter.write_dfs(ml_dict, self.ml_data_file)

    # ------ Machine Learning Task
    def fit_logistic_ols(self, ml_df_method, load_ml_df=True):
        # ------ generate data set
        if load_ml_df:
            ml_dict = H5FileSupporter.read_h5(self.ml_data_file)
        else:
            ml_dict = self.prepare_ml_df(ml_df_method)
        ml_df: pd.DataFrame = ml_dict['/df']
        f_list = ml_dict['/f_list'].values
        l_list = ml_dict['/l_list'].values

        # ------ debug
        # f_cov_matrix = ml_df[f_list].corr()
        # sns.heatmap(f_cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        # plt.show()

        # ------ preprocessing data set
        ml_df[f_list] = RobustScaler(with_centering=True, with_scaling=True).fit_transform(ml_df[f_list])
        # ml_df[l_list] = KBinsDiscretizer(n_bins=5, encode='ordinal').fit_transform(ml_df[l_list])

        # ------ fit data
        for l_tag in l_list:
            x_data = sm.add_constant(ml_df[f_list])
            y_data = ml_df[l_tag]

            model = MNLogit(endog=y_data, exog=x_data)
            res = model.fit()
            y_prob = res.predict(x_data)

            print(f'[INFO]: {l_tag}:')
            print(res.summary())
            self.print_classifier_metric(y_data.values, y_prob)
            print()


if __name__ == '__main__':
    cell = DataCell(name='dc1', cache_dir='/cell_cache')

    # ------ normalize data
    # cell.merge_contract_index(
    #     'D:/QuantPipeline/download_data/gm_contract_info.h5',
    #     'D:/QuantPipeline/download_data/gm_symbol_3600s.h5',
    #     symbol_filter=[
    #         '苹果', '玻璃', '甲醇', '纯碱', '烧碱', '白糖', 'PTA', '豆一', '豆二', '胶合板', '铁矿石', '焦炭', '焦煤',
    #         '豆粕', '塑料', 'PVC', '豆油', '沥青', '燃油', '热卷', '螺纹钢', '橡胶', '不锈钢', '线材', '郑棉', '红枣',
    #         '甲醇', '菜粕', '菜油', '菜籽', '硅铁'
    #     ],
    #     contract_filter=['连一'],
    # )
    cell.prepare_norm_data(force_update=True)
    # cell.prepare_norm_data(force_update=False)
    # for symbol, df in cell.norm_dict.items():
    #     plt.title(f'{symbol}')
    #     plt.plot(df['close'])
    #     plt.plot(df['log_close'])
    #     plt.show()

    # ------ investigate
    # cell.investigate_variance_by_length()
    # cell.investigate_statistic_by_hour()
    # cell.investigate_statistic_by_weekday()
    # cell.investigate_acorr()
    # cell.investigate_cross_section_corr()
    # cell.investigate_hour_over_hour()
    cell.investigate_objects()

    # agent.prepare_ml_df(method='base')

    # agent.fit_ols()
    # agent.fit_logistic_ols(ml_df_method='base', load_ml_df=True)
