import pandas as pd
import numpy as np
from typing import Union, Literal
from abc import ABC, abstractmethod

from utils.process_utils import get_ols_beta


class MathBaseLabeler(ABC):
    label_style: Literal['time_series', 'cross_section']

    def __init__(self, label_name: str):
        self.label_name = label_name

    @abstractmethod
    def label(self, name: str, data: pd.Series, t_events: pd.Series, **kwargs) -> pd.Series:
        '''index of t_events is start time and value is end time'''
        raise NotImplementedError

    @staticmethod
    def get_horizontal_t_events(t_events: pd.Series, price: pd.Series, num_days=0, num_hours=0, num_minutes=0, num_seconds=0) -> pd.Series:
        timedelta = pd.Timedelta('{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
        nearest_index = price.index.searchsorted(t_events + timedelta)
        nearest_index = nearest_index[nearest_index < price.shape[0]]
        nearest_timestamp = price.index[nearest_index]
        filtered_events = t_events[:nearest_index.shape[0]]
        vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
        return vertical_barriers
    

class ReturnSignLabeler(MathBaseLabeler):
    '''predict the sign of the price change'''
    label_style = 'time_series'

    def label(
            self, price: pd.Series, t_events: pd.Series, threshold: float = 0.0, window: int = None, standardize: bool = False
        ) -> pd.Series:
        rets = pd.Series(
            np.log(price.loc[t_events.values].values) - np.log(price.loc[t_events.index].values), 
            index=t_events.index
        )
        if standardize:
            ret_daily = np.log(price.shift(-1) / price)
            mean = ret_daily.rolling(window=window).mean()
            stdev = ret_daily.rolling(window=window).std()
            rets = (rets - mean) / stdev
            rets.dropna(inplace=True)
        
        labels = pd.Series(index=t_events.index)
        labels[rets.lt(-threshold)] = -1
        labels[rets.gt(threshold)] = 1
        labels[(rets.ge(-threshold)) & (rets.le(threshold))] = 0
        labels.dropna(inplace=True)
        return labels


class ScaleLabeler(MathBaseLabeler):
    '''predict the scale of the price change, meta-label'''
    label_style = 'time_series'

    def label(self, price: pd.Series, t_events: pd.Series, sign_series: pd.Series=None) -> pd.Series:
        labels = pd.Series(
            np.log(price.loc[t_events.values].values) - np.log(price.loc[t_events.index].values),
            index=t_events.index
        )
        if sign_series is not None:
            labels[sign_series != np.sign(labels)] = 0.0
        labels = labels.abs()
        labels.dropna(inplace=True)
        return labels


class TripleBarrierLabeler(MathBaseLabeler):
    '''consider the path of price and find the earliest time to hit the barrier'''
    label_style = 'time_series'

    def label(self, price: pd.Series, t_events: pd.Series, barrier_horizontal: pd.DataFrame) -> pd.DataFrame:
        """sl mean stop loss, pt mean profit taking"""
        label = pd.Series(index=t_events.index)
        for loc, vertical_date in t_events.iteritems():
            closing_prices = price.loc[loc: vertical_date]
            cum_returns = (closing_prices / price.loc[loc] - 1)
            label.at[loc, 'sl'] = cum_returns[cum_returns < barrier_horizontal.loc[loc, 'sl']].index.min()  # Earliest stop loss date
            label.at[loc, 'pt'] = cum_returns[cum_returns > barrier_horizontal.loc[loc, 'pt']].index.min()  # Earliest profit taking date
            label.at[loc, 't1'] = np.min([label.at[loc, 'sl'], vertical_date, label.at[loc, 'pt']])
            label.at[loc] = np.argmin([label.at[loc, 'sl'], vertical_date, label.at[loc, 'pt']]) - 1
        label.dropna(inplace=True)
        return label


class OverMeanLabeler(MathBaseLabeler):
    '''predict cross-section returns over mean'''
    label_style = 'cross_section'

    def label(self, prices: pd.DataFrame, t_events: pd.Series, interval: int, binary: bool = False) -> pd.DataFrame:
        labels = pd.DataFrame(index=t_events.index)
        returns = prices.pct_change(periods=interval).shift(-interval)
        market_return = returns.mean(axis=1)
        returns_over_mean = returns.sub(market_return, axis=0)
        labels['return_over_mean'] = returns_over_mean
        if binary:
            labels['return_over_mean'] = np.sign(labels['return_over_mean'])
        labels.dropna(inplace=True)
        return labels
    

class OverMedianLabeler(MathBaseLabeler):
    '''predict cross-section returns over median'''
    label_style = 'cross_section'

    def label(self, prices: pd.DataFrame, t_events: pd.Series, interval: int, binary: bool = False) -> pd.DataFrame:
        labels = pd.DataFrame(index=t_events.index)
        returns = prices.pct_change(periods=interval).shift(-interval)
        median_return = returns.median(axis=1)
        returns_over_median = returns.sub(median_return, axis=0)
        labels['return_over_median'] = returns_over_median
        if binary:
            labels['return_over_median'] = np.sign(labels['return_over_median'])
        labels.dropna(inplace=True)
        return labels
    

class TailSetLabler(MathBaseLabeler):
    '''classify the stocks into several buckets and predict the tail set'''
    label_style = 'cross_section'

    def _extract_tail_sets(self, row, n_bins: int):
        row = row.rank(method='first')  # To avoid error with unique bins when using qcut due to too many 0 values.
        row_quantiles = pd.qcut(x=row, q=n_bins, labels=range(1, 1 + n_bins), retbins=False)
        row_quantiles = row_quantiles.values  # Convert to numpy array
        row_quantiles[(row_quantiles != 1) & (row_quantiles != n_bins)] = 0
        row_quantiles[row_quantiles == 1] = -1
        row_quantiles[row_quantiles == n_bins] = 1
        row_quantiles = pd.Series(row_quantiles, index=row.index)
        return row_quantiles

    def label(self, prices: pd.DataFrame, t_events: pd.Series, n_bins: int, vol_adj: str = None, window: int = 252) -> pd.DataFrame:
        labels = pd.DataFrame(index=t_events.index)
        rets = np.log(prices).diff().dropna()
        if vol_adj is not None:
            if vol_adj == 'mean_abs_dev':
                vol = rets.abs().ewm(span=window, min_periods=window).mean()
            elif vol_adj == 'stdev':
                vol = rets.rolling(window).std()
            rets = (rets / vol).dropna()
        tail_sets = rets.dropna().apply(self._extract_tail_sets, axis=1, n_bins=n_bins)
        labels['tail_set'] = tail_sets
        labels.dropna(inplace=True)
        return labels


class TrendScanLabeler(MathBaseLabeler):
    label_style = 'time_series'

    def label(self, prices: pd.Series, t_events: pd.Series, look_forward_window: int, min_sample_length: int, step: int) -> pd.DataFrame:
        t1_array = []  # Array of label end times
        t_values_array = []  # Array of trend t-values

        for index in t_events.index:
            subset = prices.loc[index:].iloc[:look_forward_window]
            if subset.shape[0] >= look_forward_window:
                # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
                max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
                max_t_value_index = None  # Index with maximum t-value
                max_t_value = None  # Maximum t-value signed

                for forward_window in np.arange(min_sample_length, subset.shape[0], step):
                    y_subset = subset.iloc[:forward_window].values.reshape(-1, 1)  # y{t}:y_{t+l}

                    # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
                    X_subset = np.ones((y_subset.shape[0], 2))
                    X_subset[:, 1] = np.arange(y_subset.shape[0])

                    # Get regression coefficients estimates
                    b_mean_, b_std_ = get_ols_beta(X_subset, y_subset)
                    t_beta_1 = (b_mean_[1] / np.sqrt(b_std_[1, 1]))[0]
                    if abs(t_beta_1) > max_abs_t_value:
                        max_abs_t_value = abs(t_beta_1)
                        max_t_value = t_beta_1
                        max_t_value_index = forward_window
                
                # Store label information (t1, return)
                t1_array.append(subset.index[max_t_value_index - 1])
                t_values_array.append(max_t_value)
            
            else:
                t1_array.append(None)
                t_values_array.append(None)

        labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events.index)
        labels.loc[:, 'ret'] = prices.reindex(labels.t1).values / prices.reindex(labels.index).values - 1
        labels['bin'] = labels.t_value.apply(np.sign)
        labels.dropna(inplace=True)
        return labels
