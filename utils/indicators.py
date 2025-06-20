import pandas as pd
import numpy as np


class IndicatorUtils(object):
    @staticmethod
    def compute_macd(df: pd.DataFrame, inplace=True):
        short = df['close'].ewm(span=12).mean()
        long = df['close'].ewm(span=26).mean()
        if inplace:
            df['dif'] = short - long
            df['dea'] = df['dif'].ewm(span=9).mean()
            df['macd'] = (df['dif'] - df['dea']) * 2.0
        else:
            return pd.DataFrame({
                'dif': short - long,
                'dea': df['dif'].ewm(span=9).mean(),
                'macd': (df['dif'] - df['dea']) * 2.0
            })
        return df

    @staticmethod
    def compute_kdj(df: pd.DataFrame, inplace=True):
        lowest = df['close'].rolling(9).min()
        highest = df['close'].rolling(9).max()
        rsv = (df['close'] - lowest) / (highest - lowest) * 100.0
        if inplace:
            df['kdj_k'] = rsv.ewm(com=2).mean()
            df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
            df['kdj_j'] = 3.0 * df['kdj_k'] - 2.0 * df['kdj_d']
        else:
            return pd.DataFrame({
                'kdj_k': rsv.ewm(com=2).mean(),
                'kdj_d': df['kdj_k'].ewm(com=2).mean(),
                'kdj_j': 3.0 * df['kdj_k'] - 2.0 * df['kdj_d']
            })
        return df

    @staticmethod
    def compute_ma(df: pd.DataFrame, periods: list[int], inplace=True):
        if inplace:
            for period in periods:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
        else:
            return pd.DataFrame({
                f'ma_{period}': df['close'].rolling(period).mean()
                for period in periods
            })
        return df