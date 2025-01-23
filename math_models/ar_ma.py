from typing import List, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.base.prediction import PredictionResults

import pmdarima
from pmdarima.arima import auto_arima, ADFTest


class TimeSeriesTest(object):
    @staticmethod
    def acf(x: np.ndarray, nlags: int, alpha=None, q_stat=False, only_plot=False):
        """
        Calculate the auto-correlation
        alpha: If a number is given, the confidence intervals for the given level are returned
        """
        if only_plot:
            plot_acf(x, lags=nlags, alpha=alpha)
            plt.show()
        else:
            return sm.tsa.acf(x=x, nlags=nlags, alpha=alpha, qstat=q_stat)

    def pacf(self, x: np.ndarray, nlags: int, alpha=None, only_plot=False, method='ywadjusted'):
        """Partial autocorrelation estimate: 假设求解 t与t+k 之间的相关系数，先基于例如 OLS 逐级剔除 t与t+1，t+2的关系"""
        if only_plot:
            plot_pacf(x=x, lags=nlags, alpha=alpha, method=method)
            plt.show()
        else:
            return sm.tsa.pacf(x=x, nlags=nlags, alpha=alpha, method=method)

    @staticmethod
    def adfuller(
            x: np.ndarray, maxlag: int = None,
            trend: Literal["c", "ct", "ctt", "n"] = 'c', autolag: Literal['AIC', "BIC", "t-stat"] = 'AIC'
    ):
        """
        Augmented Dickey-Fuller unit root test：: test whether series is stable
        H0: there is root, H1: there is no root
        (1.0 - p-value): probability to reject H0
        """
        res = sm.tsa.adfuller(x=x, maxlag=maxlag, regression=trend, autolag=autolag)
        return {'t_value': res[0], 'pvalue': res[1], 'critical_t_values': res[4]}

    @staticmethod
    def breakvar_heteroskedasticity_test(
            residual: np.ndarray, subset_length=0.35,
            alternative: Literal['increasing', 'decreasing', 'two-sided'] = 'two-sided'
    ):
        """
        检验残差的异方差性: 逐步增长序列检验方差是否变动
        H0: that the variance is not changing
        """
        res = sm.tsa.breakvar_heteroskedasticity_test(residual, subset_length)
        return {'t_value': res[0], 'p_value': res[1]}

    @staticmethod
    def arma_order_select_ic(
            x: np.ndarray, max_ar=4, max_ma=4, ic=["aic", "bic"], trend: Literal["c", "n"] = 'n'
    ):
        res = sm.tsa.arma_order_select_ic(x, max_ar, max_ma, ic, trend=trend)
        return {'aic_min_order': res.aic_min_order, 'bic_min_order': res.bic_min_order}

    @staticmethod
    def find_integration_order(x: np.ndarray, max_order=10) -> (int, bool):
        x_series = pd.Series(x)
        adf_test_runner = ADFTest(alpha=0.05)
        diff_order, cur_order, is_stationary_flag = 0, 0, False
        while not is_stationary_flag:
            test_series = x_series.copy()
            for _ in range(cur_order):
                test_series = test_series.diff().dropna()
            if not adf_test_runner.should_diff(test_series)[1]:
                diff_order = cur_order
                is_stationary_flag = True
            if cur_order >= max_order:
                break
            cur_order += 1
        return diff_order, is_stationary_flag


class ARIMAModel(object):
    def __init__(self):
        self.model: ARIMA = None
        self.model_res: ARIMAResults = None

    def train(
            self, y_train: np.ndarray, order_p: int, order_d: int, order_q: int,
            regressor_X: np.ndarray=None, trend: Literal['n', 'c', 't', 'ct'] = 'n'
    ):
        """
        trend : {'n', 'c', 't', 'ct'}
        The trend to include in the model:
        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.
        """
        self.model = ARIMA(endog=y_train, exog=regressor_X, order=(order_p, order_d, order_q), trend=trend)
        self.model_res = self.model.fit()
        return self.model_res

    @property
    def params(self):
        return self.model_res.params

    @property
    def cov_params(self):
        return self.model_res.cov_params()

    @property
    def summary(self, alpha=0.05):
        """
        :param alpha: Significance level for the confidence intervals
        """
        return self.model_res.summary(alpha=alpha)

    def predict(self, start: int, end: int, dynamic=False, regressor_X: np.ndarray=None):
        """In Sample: dynamic=False means use actual value, dynamic=True mean based on predict value"""
        return self.model_res.predict(start, end, dynamic=dynamic, exog=regressor_X)

    def forecast(self, steps: int, regressor_X: np.ndarray=None):
        """Out of Sample"""
        return self.model_res.forecast(steps, exog=regressor_X)

    def get_predict_dist(self, start: int, end: int, dynamic=False, regressor_X: np.ndarray=None):
        dist: PredictionResults = self.model_res.get_prediction(start, end, dynamic=dynamic, exog=regressor_X)
        return {'means': dist.predicted_mean, 'vars': dist.var_pred_mean}

    def get_forest_dist(self, steps: int, regressor_X: np.ndarray=None):
        dist: PredictionResults = self.model_res.get_forecast(steps, exog=regressor_X)
        return {'means': dist.predicted_mean, 'vars': dist.var_pred_mean}

    def simulate(self, steps: int):
        return self.model.simulate(self.params, steps)

    def append(self, new_x: np.ndarray, regressor_X: np.ndarray=None) -> ARIMAResults:
        """append new train data: 增量训练"""
        return self.model_res.append(new_x, refit=False, exog=regressor_X)

    def apply(self, new_x: np.ndarray, regressor_X: np.ndarray=None) -> ARIMAResults:
        """
        Apply the fitted parameters to new data unrelated to the original data
        返回一个结果wrapper,基于该wrapper进行forecast
        """
        return self.model_res.apply(new_x, refit=False, exog=regressor_X)


class AutoARIMAModel(object):
    def __init__(self):
        self.model: pmdarima.arima.ARIMA = None

    def train(
            self,
            y_train: np.ndarray, regressor_X: np.ndarray=None,
            start_p=0, start_q=0, max_p=5, max_q=5, max_d=10,
            season_period=1, with_seasonal=False,
            information_criterion: Literal['aic', 'bic', 'hqic', 'oob']='aic',
            trend: Literal['n', 'c', 't', 'ct', None] = None,
            verbose: bool = False, order_d=None, n_jobs=1
    ):
        order_d, success_flag = TimeSeriesTest.find_integration_order(y_train)
        if not success_flag:
            return False

        if trend is None:
            with_intercept = 'auto'
        elif trend == 'n':
            with_intercept = False
        else:
            with_intercept = True

        self.model = auto_arima(
            y=y_train,
            X=regressor_X,
            d=order_d,
            start_p=start_p,
            start_q=start_q,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            m=season_period,
            seasonal=with_seasonal,
            information_criterion=information_criterion,
            alpha=0.05,
            test='kpss',  # use for order d determine
            seasonal_test='ocsb',
            trend=trend,
            with_intercept=with_intercept,
            n_jobs=n_jobs,
            trace=verbose,
        )
        return True

    @property
    def params(self):
        return self.model.params

    @property
    def summary(self):
        return self.model.summary()

    def predict(self, steps: int, return_conf_int=False, alpha=0.05, regressor_X: np.ndarray=None):
        return self.model.predict(steps, return_conf_int=return_conf_int, alpha=alpha, X=regressor_X)

    def append(self, y_train: np.ndarray, regressor_X: np.ndarray=None, max_iter=None):
        """Updating an ARIMA adds new observations to the model: 增量训练"""
        self.model = self.model.update(y_train, regressor_X, maxiter=max_iter)

    def retrain_predict(self, y_train: np.ndarray, n_periods: int, regressor_X: np.ndarray=None):
        return self.model.fit_predict(y_train, regressor_X, n_periods)


# data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
# n_data = data.shape[0]
# exog_data = np.random.random(size=(n_data, 5))
