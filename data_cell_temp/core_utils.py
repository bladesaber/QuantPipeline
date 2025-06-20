import os
import pandas as pd
import numpy as np
from numpy import random
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score


class IndicatorUtils(object):
    @staticmethod
    def compute_macd(df: pd.DataFrame):
        short = df['close'].ewm(span=12).mean()
        long = df['close'].ewm(span=26).mean()
        df['dif'] = short - long
        df['dea'] = df['dif'].ewm(span=9).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2.0
        return df

    @staticmethod
    def compute_kdj(df: pd.DataFrame):
        lowest = df['close'].rolling(9).min()
        highest = df['close'].rolling(9).max()
        rsv = (df['close'] - lowest) / (highest - lowest) * 100.0
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3.0 * df['kdj_k'] - 2.0 * df['kdj_d']
        return df

    @staticmethod
    def compute_ma(df: pd.DataFrame):
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_40'] = df['close'].rolling(40).mean()
        df['ma_60'] = df['close'].rolling(60).mean()
        df['ma_120'] = df['close'].rolling(120).mean()
        return df


class DataCellUtils(object):
    @staticmethod
    def print_classifier_metric(y_true: np.ndarray, y_prob: np.ndarray):
        labels = np.unique(y_true)
        y_predict = np.argmax(y_prob, axis=1)
        precision_list = precision_score(y_true, y_predict, average=None, labels=labels)
        recall_list = recall_score(y_true, y_predict, average=None, labels=labels)
        for i, label in enumerate(labels):
            print(f"Label:{int(label)} Precision:{float(precision_list[i])} Recall:{float(recall_list[i])}")

    @staticmethod
    def print_label_description(df: pd.DataFrame, label_tag:str):
        print(f"[INFO]: Label:{label_tag}")
        for (label

            ), count in df[label_tag].value_counts().items():
            print(f"    {label}:{count} percent:{count/df.shape[0]}")
