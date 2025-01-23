from abc import abstractmethod
from typing import List, Union, Literal, Callable
import numpy as np
import pandas as pd


class Downloader(object):
    @abstractmethod
    def download_bar(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def download_tick(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def save_hdf5(data: pd.DataFrame, filepath: str, name: str):
        data.to_hdf(filepath, key=name)

    @staticmethod
    def read_hdf5(filepath: str) -> pd.DataFrame:
        return pd.read_hdf(filepath)

    @staticmethod
    def save_csv(data: pd.DataFrame, filepath: str):
        data.to_csv(filepath)

    @staticmethod
    def read_csv(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)


class PreprocessLoader(object):
    @staticmethod
    def apply_func(data: pd.DataFrame, key: str, func: Callable) -> pd.Series:
        return data[key].apply(lambda x: func(x))

    @staticmethod
    def convert_timestep(data: pd.DataFrame, keys: List[str], mode='%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
        for key in keys:
            data[key] = pd.to_datetime(data[key], format=mode)
        return data

    @staticmethod
    def convert_week_day(data: pd.DataFrame, key: str) -> pd.Series:
        return data[key].dt.weekday

    @staticmethod
    def convert_day(data: pd.DataFrame, key: str) -> pd.Series:
        return data[key].dt.day

    @staticmethod
    def convert_hour(data: pd.DataFrame, key: str) -> pd.Series:
        return data[key].dt.hour

    @staticmethod
    def resample_date(
            data: pd.DataFrame, date_key: str, mode='1D',
            func: Union[Callable, Literal['mean', 'max', 'min']] = 'mean'
    ):
        new_data = data.set_index(keys=[date_key])
        new_data = new_data.resample(mode).aggregate(func=func)
        new_data.reset_index(inplace=True)
        return new_data

    @staticmethod
    def norm_data(data: pd.DataFrame, date_key: str, OLHC_tags: List[str], vol_tag: str = None):
        data.sort_values(by=[date_key], inplace=True)
        norm_data = pd.DataFrame({date_key: data[date_key]})
        norm_data[OLHC_tags[3]] = data[OLHC_tags[3]] / data.iloc[0][OLHC_tags[3]]
        for tag in OLHC_tags:
            norm_data[f'r_{tag}'] = data[tag] / data[OLHC_tags[3]].shift(1) - 1.0
        if vol_tag is not None:
            norm_data['r_vol'] = data[vol_tag] / data[vol_tag].shift(1) - 1.0
        return norm_data


