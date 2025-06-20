import numpy as np
import pandas as pd

from utils.process_utils import Utils


class MicroStructuralUtils:
    @staticmethod
    def bar_based_kyle_lambda(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Advances in Financial Machine Learning, p.288-289.
        Get Amihud lambda from bars data
        """
        sign_diff = price.diff()
        sign_diff_sign = sign_diff.apply(np.sign)
        sign_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
        volume_mult_trade_signs = volume * sign_diff_sign  # bt * Vt
        return (sign_diff / volume_mult_trade_signs).rolling(window=window).mean()

    @staticmethod
    def bar_based_amihud_lambda(price: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Advances in Financial Machine Learning, p.288-289.
        Get Amihud lambda from bars data
        """
        returns_abs = np.log(price / price.shift(1)).abs()
        return (returns_abs / np.sqrt(dollar_volume)).rolling(window=window).mean()

    @staticmethod
    def trades_based_kyle_lambda(price: pd.Series, volume: pd.Series) -> pd.Series:
        y = price.diff().values
        X = volume * np.sign(y)
        beta, beta_var = Utils.get_ols_beta(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
        t_value = beta[0] / np.sqrt(beta_var[0])
        return [beta[0], t_value]
    
    @staticmethod
    def trades_based_hasbrouck_lambda(price: pd.Series, dollar_volume: pd.Series) -> pd.Series:
        y = np.log(price / price.shift(1)).abs().values
        X = dollar_volume.values
        beta, beta_var = Utils.get_ols_beta(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
        t_value = beta[0] / np.sqrt(beta_var[0])
        return [beta[0], t_value]
    
    @staticmethod
    def get_VPIN():
        raise NotImplementedError("VPIN is not implemented yet")
    
    @staticmethod
    def get_PIN():
        raise NotImplementedError("PIN is not implemented yet")


def extract_bars_features(data: pd.DataFrame, bar_time_series: pd.Series) -> pd.DataFrame:
    """ Just an example, not used in the project """
    tick_num_generator = iter(bar_time_series)
    current_bar_time = next(tick_num_generator)
    feat_df = {}
    
    bars = []
    for row in data.itertuples():
        bars.append(row)
        if row.Index > current_bar_time:
            feat_df[current_bar_time] = {
                'kyle_lambda': MicroStructuralUtils.bar_based_kyle_lambda(bars),
                'amihud_lambda': MicroStructuralUtils.bar_based_amihud_lambda(bars),
                'kyle_lambda_t_value': MicroStructuralUtils.trades_based_kyle_lambda(bars),
                'hasbrouck_lambda_t_value': MicroStructuralUtils.trades_based_hasbrouck_lambda(bars)
            }
            bars = []
            current_bar_time = next(tick_num_generator)

    return pd.DataFrame(feat_df, index=list(feat_df.keys()))