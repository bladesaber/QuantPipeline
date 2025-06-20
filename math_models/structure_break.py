from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

import ruptures as rpt  # package for change point detection
import pytimetk         # package for easy time series analysis not research level, good for visualization
import merlion          # https://github.com/salesforce/Merlion machine learning for time series
import darts            # https://github.com/unit8co/darts time series analysis

'''
1.statical hypothesis test for time series
2.entropy threshold for time series
3.general model for time series and test its residual
4.find all structure breaks based on whole time series and learn an model to predict it
5.market strucutre model and test its probability
6.statical hypothesis test for label time series
传统的structure break指代从稳定状态到不稳定状态的转变
'''

class MathBaseStructureBreakModel(ABC):
    detection_style: Literal['unstable_detection', 'value_shift_detection']
    stream_style: Literal['online', 'offline']

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def fit(self, signal: np.ndarray):
        pass


class ChowTest(MathBaseStructureBreakModel):
    '''
    https://en.wikipedia.org/wiki/Chow_test
    classical:only used in when break point is known and only one break point
    Quandt-Andrews Test:Tests for a single structural break when the exact timing is unknown
    only accept one dimension data
    '''

    method: Literal['classical', 'Quandt-Andrews Test', 'Bai-Perron Test']

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'offline'
        self.results = None

    def classical(self, y_pred0: np.ndarray, y_pred1: np.ndarray, y_true0: np.ndarray, y_true1: np.ndarray, k: int):
        """
        Classical Chow test for known break point.
        
        Args:
            y_pred0: Predictions before break point
            y_pred1: Predictions after break point
            y_true0: True values before break point
            y_true1: True values after break point
            k: number of parameters of model e.g intercept + slope in linear regression
            
        Returns:
            dict: Test results including F-statistic and p-value
        """
        # Calculate residuals for each period
        residuals0 = y_true0 - y_pred0
        residuals1 = y_true1 - y_pred1

        # Calculate sum of squared residuals for each period
        SSR0 = np.sum(residuals0**2)
        SSR1 = np.sum(residuals1**2)

        # Calculate combined residuals (pooled model)
        y_combined = np.concatenate([y_true0, y_true1])
        y_pred_combined = np.concatenate([y_pred0, y_pred1])
        residuals_combined = y_combined - y_pred_combined
        SSR_combined = np.sum(residuals_combined**2)

        # Number of observations and parameters
        n0 = len(y_pred0)
        n1 = len(y_pred1)
        n = n0 + n1

        # Calculate Chow test statistic
        numerator = (SSR_combined - (SSR0 + SSR1)) / k
        denominator = (SSR0 + SSR1) / (n - 2*k)
        F_stat = numerator / denominator

        # Calculate p-value using F-distribution
        p_value = 1 - f_distribution.cdf(F_stat, k, n - 2*k)

        self.results = {
            'F_statistic': F_stat,
            'p_value': p_value,
            'SSR0': SSR0,
            'SSR1': SSR1,
            'SSR_combined': SSR_combined,
            'n0': n0,
            'n1': n1,
            'k': k
        }
        return self.results

    def Quandt_Andrews_Test(self, y_pred: np.ndarray, y_true: np.ndarray, min_size: int = 15, trim: float = 0.15):
        """
        Quandt-Andrews test for unknown break point.
        
        Args:
            y_pred: All predictions
            y_true: All true values
            min_size: Minimum size of each segment
            trim: Percentage of data to trim from each end
            
        Returns:
            dict: Test results including max F-statistic and break point
        """
        n = len(y_pred)
        trim_size = int(n * trim)
        possible_breaks = range(trim_size, n - trim_size)
        
        max_F = -np.inf
        best_break = None
        
        for break_point in possible_breaks:
            # Split data at current break point
            y_pred0 = y_pred[:break_point]
            y_pred1 = y_pred[break_point:]
            y_true0 = y_true[:break_point]
            y_true1 = y_true[break_point:]
            
            # Skip if segments are too small
            if len(y_pred0) < min_size or len(y_pred1) < min_size:
                continue
                
            # Perform classical Chow test
            results = self.classical(y_pred0, y_pred1, y_true0, y_true1)
            
            if results['F_statistic'] > max_F:
                max_F = results['F_statistic']
                best_break = break_point
        
        # Calculate p-value for max F-statistic
        p_value = 1 - f_distribution.cdf(max_F, 1, n - 2)
        
        self.results = {
            'max_F_statistic': max_F,
            'p_value': p_value,
            'break_point': best_break,
            'n': n
        }
        return self.results

    def fit(self, signal_pred: np.ndarray, signal_true: np.ndarray):
        return self.Quandt_Andrews_Test(signal_pred, signal_true)


class RupturesModel(MathBaseStructureBreakModel):
    '''
    https://centre-borelli.github.io/ruptures-docs/user-guide/
    Object: Finds the (exact) minimum of the sum of costs by computing the cost of all subsequences of a given signal series
    Dynamic programming:search over all possible segmentations
    Linearly penalized segmentation:Because the enumeration of all possible partitions impossible, 
        the algorithm relies on a pruning rule. Many indexes are discarded, greatly reducing the computational cost
    Window-based:Using two windows which slide along the data stream
    Kernel change point detection:Kernel method computes similarity/distance matrix between each pair of points, 
        then uses the matrix to detect change points
    multiple dimension data allowed
    '''
    method: Literal[
        'Dynamic programming', 'Linearly penalized segmentation', 'Binary segmentation', 
        'Bottom-up segmentation', 'Kernel change point detection'
    ]
    cost: Literal['l1', 'l2', 'l1_l2'] = 'l2'
    kernel: Literal['rbf', 'linear', 'cosine'] = 'rbf'

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'offline'
        self.results = None

    @staticmethod
    def generate_sample(n_samples: int, dim: int, n_bkps: int, noise_std: float):
        signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=noise_std)
        return signal, bkps

    def fit(
        self, 
        signal: np.ndarray,
        min_size: int = 5,
        n_bkps: int = None,
        jumps: int =5,
        pelt_penalty: float = None,
        method: Literal[
            'Dynamic programming', 'Linearly penalized segmentation', 'Binary segmentation', 
            'Bottom-up segmentation', 'Kernel change point detection'
        ] = 'Linearly penalized segmentation',
        cost: Literal['l1', 'l2', 'normal', 'rbf', 'cosine', 'linear'] = 'l2',
        kernel: Literal['rbf', 'linear', 'cosine'] = 'rbf',
        **kwargs
    ):
        '''
        param:
            n_bkps: number of break points
            jumps: subsample (one every *jump* points)
            pelt_penalty: penalty for linearly penalized segmentation
        '''
        if method == 'Dynamic programming':
            assert n_bkps is not None, "jumps must be provided for dynamic programming"
            model = rpt.Dynp(model=cost, min_size=min_size, jumps=jumps).fit(signal)
            self.results = model.predict(n_bkps=n_bkps)

        elif method == 'Linearly penalized segmentation':
            assert pelt_penalty is not None, "pelt_penalty must be provided for linearly penalized segmentation"
            model = rpt.Pelt(model=cost, min_size=min_size, jumps=jumps).fit(signal)
            self.results = model.predict(pen=pelt_penalty)

        elif method == 'Binary segmentation':
            model = rpt.Binseg(model=cost, min_size=min_size, jumps=jumps).fit(signal)
            if n_bkps is not None:
                self.results = model.predict(n_bkps=n_bkps)
            else:
                noise_std = np.std(signal)
                self.results = model.predict(n_bkps=np.log(signal.shape[0]) * signal.shape[1] * noise_std**2)

        elif method == 'Bottom-up segmentation':
            model = rpt.BottomUp(model=cost, min_size=min_size, jumps=jumps).fit(signal)
            if n_bkps is not None:
                self.results = model.predict(n_bkps=n_bkps)
            else:
                noise_std = np.std(signal)
                self.results = model.predict(n_bkps=np.log(signal.shape[0]) * signal.shape[1] * noise_std**2)

        elif method == 'Kernel change point detection':
            model = rpt.KernelCPD(model=kernel, min_size=min_size, jumps=jumps).fit(signal)
            if n_bkps is not None:
                self.results = model.predict(n_bkps=n_bkps)
            else:
                self.results = model.predict(pen=pelt_penalty)

        else:
            raise ValueError(f"Invalid method: {method}")
        
        return self.results


class CumsumModel(MathBaseStructureBreakModel):
    """
    CUSUM (Cumulative Sum) model for structural break detection.
    This model is particularly useful for online detection of structural breaks.
    only accept one dimension data
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'online'
        self.results = None
    
    def positive_fit(
            self, 
            residual: pd.Series, window_size: int = 30, threshold: float = 2.0, 
            mean_shift: bool = True, step_process: bool = True
        ):
        pos_df = pd.DataFrame({'residual': residual.values}, index=residual.index)
        if mean_shift:
            pos_df['rolling_mean'] = pos_df['residual'].rolling(window=window_size, min_periods=1).mean()
            pos_df['residual'] = pos_df['residual'] - pos_df['rolling_mean']

        pos_df['rolling_std'] = pos_df['residual'].rolling(window=window_size, min_periods=1).std()

        for t in range(window_size, len(pos_df)):
            epsilon = pos_df['residual'].iloc[t]
            rolling_std = pos_df['rolling_std'].iloc[t]
            last_S_pos = pos_df['S_pos'].iloc[t-1]

            pos_df['S_pos'].iloc[t] = max(0, last_S_pos + epsilon) / rolling_std
            if pos_df['S_pos'].iloc[t] > threshold:
                pos_df.loc[pos_df.index[t], 'pos_break_points'] = True
                if step_process:
                    pos_df['S_pos'].iloc[t] = 0
        
        return pos_df

    def negative_fit(
            self, 
            residual: pd.Series, window_size: int = 30, threshold: float = 2.0, 
            mean_shift: bool = True, step_process: bool = True
        ):
        neg_df = pd.DataFrame({'residual': residual.values}, index=residual.index)
        if mean_shift:
            neg_df['rolling_mean'] = neg_df['residual'].rolling(window=window_size, min_periods=1).mean()
            neg_df['residual'] = neg_df['residual'] - neg_df['rolling_mean']

        neg_df['rolling_std'] = neg_df['residual'].rolling(window=window_size, min_periods=1).std()

        for t in range(window_size, len(neg_df)):
            epsilon = neg_df['residual'].iloc[t]
            rolling_std = neg_df['rolling_std'].iloc[t]
            last_S_neg = neg_df['S_neg'].iloc[t-1]

            neg_df['S_neg'].iloc[t] = min(0, last_S_neg + epsilon) / rolling_std
            if neg_df['S_neg'].iloc[t] < -threshold:
                neg_df.loc[neg_df.index[t], 'neg_break_points'] = True
                if step_process:
                    neg_df['S_neg'].iloc[t] = 0
        
        return neg_df

    def fit(self, residual: pd.Series, window_size: int = 30, threshold: float = 3.0, 
            mean_shift: bool = True, step_process: bool = True, with_positive: bool = True, with_negative: bool = True
        ):
        if with_positive:
            pos_df = self.positive_fit(residual, window_size, threshold, mean_shift, step_process)
        if with_negative:
            neg_df = self.negative_fit(residual, window_size, threshold, mean_shift, step_process)
        
        self.results = {
            'pos_break_points_df': pos_df,
            'neg_break_points_df': neg_df,
            'window_size': window_size,
            'threshold': threshold,
            'mean_shift': mean_shift,
            'step_process': step_process
        }
        
        return self.results


class WindowSlideModel(MathBaseStructureBreakModel):
    '''
    https://centre-borelli.github.io/ruptures-docs/user-guide/
    multiple dimension data allowed
    '''
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'online'
        self.results = None
    
    def fit(
        self, 
        signal: pd.DataFrame,
        min_size: int = 5,
        n_bkps: int = None,
        jumps: int =5,
        pelt_penalty: float = None,
        cost: Literal['l1', 'l2', 'normal', 'rbf', 'cosine', 'linear'] = 'l2',
    ):
        model = rpt.Window(model=cost, min_size=min_size, jumps=jumps).fit(signal.values)
        if n_bkps is not None:
            self.results = model.predict(n_bkps=n_bkps)
        elif pelt_penalty is not None:
            self.results = model.predict(pen=pelt_penalty)
        else:
            noise_std = np.std(signal.values)
            self.results = model.predict(n_bkps=np.log(signal.shape[0]) * signal.shape[1] * noise_std**2)
        return self.results


class ZScoreModel(MathBaseStructureBreakModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'online'
        self.results = None
    
    def fit(self, signal: pd.Series, window_size: int = 30, threshold: float = 3.0):
        rolling_mean = signal.rolling(window=window_size, min_periods=1).mean()
        rolling_std = signal.rolling(window=window_size, min_periods=1).std()
        z_score = (signal - rolling_mean) / rolling_std
        df = pd.DataFrame({
            'signal': signal.values,
            'z_score': z_score.values,
        }, index=z_score.index)
        df['break_point'] = df['z_score'] > threshold
        return df


class LongShortDifferenceModel(MathBaseStructureBreakModel):
    '''only accept one dimension data'''
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'online'
        self.results = None

    def fit(self, signal: pd.Series, long_period: int = 30, short_period: int = 5):
        long_signal = signal.rolling(window=long_period, min_periods=1).mean()
        short_signal = signal.rolling(window=short_period, min_periods=1).mean()
        meta_signal = long_signal - short_signal
        raise NotImplementedError
        
        
class LocalRegressionModel(MathBaseStructureBreakModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.detection_style = 'value_shift_detection'
        self.stream_style = 'online'
        self.results = None

    def fit(self, signal: pd.Series, window_size: int = 30, threshold: float = 3.0):
        raise NotImplementedError
