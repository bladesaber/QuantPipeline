import pandas as pd
import numpy as np
import datetime
from typing import Union, Tuple


class TimeUtils(object):
    @staticmethod
    def datetime_to_str(dt: datetime.datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def str_to_datetime(dt_str: str):
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def convert_day(data: pd.DataFrame, key: str, inplace=True) -> pd.Series:
        if inplace:
            data[key] = data[key].dt.day
            return data[key]
        else:
            return data[key].dt.day
    
    @staticmethod
    def convert_day(data: pd.DataFrame, key: str, inplace=True) -> pd.Series:
        if inplace:
            data[key] = data[key].dt.day
            return data[key]
        else:
            return data[key].dt.day

    @staticmethod
    def convert_hour(data: pd.DataFrame, key: str, inplace=True) -> pd.Series:
        if inplace:
            data[key] = data[key].dt.hour
            return data[key]
        else:
            return data[key].dt.hour
    

class Utils(object):
    @staticmethod
    def resample_data(data: pd.DataFrame, freq: str):
        return data.resample(freq).last()

    @staticmethod
    def pct_change(data: pd.DataFrame, periods: int = 1):
        return data.pct_change(periods=periods)
        
    @staticmethod
    def pd_index_searchsorted(series: Union[pd.Series, pd.DataFrame], value: Union[float, pd.Series]) -> np.intp:
        if isinstance(value, pd.Series):
            return series.index.searchsorted(value.index)
        else:
            return series.index.searchsorted(value)

    @staticmethod
    def pd_value_searchsorted(series: Union[pd.Series, pd.DataFrame], value: Union[float, pd.Series]) -> np.intp:
        if isinstance(value, pd.Series):
            return series.searchsorted(value)
        else:
            return series.searchsorted(value)

    @staticmethod
    def get_ols_beta(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.array, np.array]:
        '''compute the beta of oridinal least square regression'''
        xy = np.dot(X.T, y)
        xx = np.dot(X.T, X)

        try:
            xx_inv = np.linalg.inv(xx)
        except np.linalg.LinAlgError:
            return [np.nan], [[np.nan, np.nan]]

        b_mean = np.dot(xx_inv, xy)
        err = y - np.dot(X, b_mean)
        b_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1])
        b_std = np.sqrt(np.diag(b_var * xx_inv))

        return b_mean, b_std