import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


def get_frac_diff_weights(diff_amt, size, thresh):
    weights = [1.]  # create an empty list and initialize the first element with 1.
    for k in range(1, size):
        weights_ = -weights[-1] * (diff_amt - k + 1) / k  # compute the next weight
        if abs(weights_) < thresh:
            break
        weights.append(weights_)
        if len(weights) >= size:
            break
    weights = np.array(weights) # from near to far
    return weights


class FeatureUtils:
    @staticmethod
    def transformation_residuals(y: np.ndarray, X: np.ndarray, with_constant: bool = True) -> np.ndarray:
        if with_constant:
            X = sm.add_constant(X)
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        return residuals
        
    @staticmethod
    def transformation_frac_diff(x: pd.Series, diff_amt: float, size: int) -> pd.Series:
        """
        Fractional difference of a time series.
        x: pd.Series from near to far
        """
        weights = get_frac_diff_weights(diff_amt, size)
        frac_values = np.convolve(x, weights, mode='full')[:len(x)]  # convolve will reverse the weights
        return pd.Series(frac_values, index=x.index)


def plot_min_ffd(series: pd.Series, thresh: float = 0.01):
    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% \conf', 'corr'])
    for d_value in np.linspace(0, 1, 11):
        differenced_series = FeatureUtils.transformation_frac_diff(
            series, diff_amt=d_value, thresh=thresh
        ).dropna()
        corr = np.corrcoef(
            series.loc[differenced_series.index].values,
            differenced_series.values
        )[0, 1]
        adf, pval, lags, nobs, crit_values, icbest = adfuller(differenced_series, maxlag=1, regression='c', autolag=None)
        results.loc[d_value] = [adf, pval, lags, nobs, crit_values['5%'], corr]
    plot = results[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(results['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.show()



