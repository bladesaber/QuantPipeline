import numpy as np
import pandas as pd


class DataCleaner(object):
    @staticmethod
    def clean_extreme_by_quantile(prices: pd.Series, quantile_range: tuple[float, float] = (0.01, 0.99)):
        min_value = prices.quantile(quantile_range[0])
        max_value = prices.quantile(quantile_range[1])
        prices[prices >= min_value] = min_value
        prices[prices <= max_value] = max_value
        return prices
    
    @staticmethod
    def clean_extreme_by_range(prices: pd.Series, range_threshold: tuple[float, float]):
        prices[prices >= range_threshold[0]] = range_threshold[0]
        prices[prices <= range_threshold[1]] = range_threshold[1]
        return prices

    @staticmethod
    def price_diff(prices: pd.Series):
        return prices.diff()
    
    @staticmethod
    def log_price(prices: pd.Series):
        return np.log(prices)
    
    @staticmethod
    def log_price_diff(prices: pd.Series):
        return np.log(prices).diff()
    
    @staticmethod
    def zscore_log_price_diff(prices: pd.Series):
        diff_values = DataCleaner.log_price_diff(prices).dropna().values
        std_value = np.std(diff_values)
        mean_value = np.mean(diff_values)
        z_scores = np.abs(diff_values - mean_value) / std_value
        return z_scores

    @staticmethod
    def align_price(prices: pd.Series):
        return prices / prices.iloc[0]
    
    @staticmethod
    def standardize_data(data: pd.Series):
        return (data - np.mean(data)) / np.std(data)
    
    @staticmethod
    def consist_log_price(prices: pd.Series, variance: float = 0.01):
        prices = DataCleaner.align_price(prices)
        zscore_log_price_diff = DataCleaner.zscore_log_price_diff(prices)
        consist_log_price_diff = zscore_log_price_diff * variance
        consist_log_price = np.cumsum(consist_log_price_diff)
        return consist_log_price
    