import os
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style, init


init(autoreset=True)


class FuturePerformanceStat(object):
    def __init__(
            self, event_df: pd.DataFrame, info_df: pd.DataFrame, bar_df: pd.DataFrame, 
            benchmark: pd.Series, asset_inital_value: float, output_dir: str
        ):
        '''
        event_df: pd.DataFrame: date_time, order_direction, volume, price, 
            order_cash, order_asset, profit, order_id, order_style, order_trade_cost
        info_df: pd.DataFrame: order_id, date_time, close_date_time, 
            order_direction, price, volume, order_style,
            profit, stop_loss_price, take_profit_price, 
            slippage_cost, context_cost, deposit, status, msg
        '''
        self.event_df: pd.DataFrame = event_df
        self.info_df: pd.DataFrame = info_df
        self.bar_df: pd.DataFrame = bar_df
        self.benchmark: pd.Series = benchmark
        self.asset_inital_value: float = asset_inital_value
        self.output_dir: str = output_dir
        
    def summary(self, with_detail: bool=False, debug: bool=False):
        self.event_df.sort_index(inplace=True)
        porfolio_df = pd.DataFrame(index=self.bar_df.index)

        for direction in ['long', 'short']:
            _event_df = self.event_df[self.event_df['order_direction'] == direction]
            
            porfolio_df[f'{direction}_volume'] = _event_df['volume']
            porfolio_df[f'{direction}_position'] = porfolio_df[f'{direction}_volume'].cumsum()

            # -------- debug --------
            if debug:
                cum_position_list = []
                profit_list = []
            # -------- debug --------

            porfolio_df[f'{direction}_avg_price'] = np.nan
            cum_position, avg_price = 0, 0.0
            for index, row in _event_df.iterrows():
                if row.order_style == 'open':
                    avg_price = (avg_price * cum_position + row['price'] * row['volume']) / (cum_position + row['volume'])
                    cum_position += row['volume']
                elif row.order_style == 'close':
                    # ------ don't update avg_price, assume rest volume all at avg_price ------
                    cum_position -= row['volume']
                else:
                    raise ValueError(f'Invalid order style: {row.order_style}')
                porfolio_df.loc[index, f'{direction}_avg_price'] = avg_price

                # -------- debug --------
                if debug:
                    cum_position_list.append({'date_time': index, 'cum_position': cum_position})
                    if row.order_style == 'close':
                        if row.order_direction == 'long':
                            profit = (self.bar_df.loc[index, 'price'] - avg_price) * row['volume']
                        else:
                            profit = (avg_price - self.bar_df.loc[index, 'price']) * row['volume']
                        profit_list.append({'date_time': index, 'profit': profit})
                # -------- debug --------
            
            # -------- debug --------
            if debug:
                cum_position_df = pd.DataFrame(cum_position_list)
                cum_position_df.set_index('date_time', inplace=True)

                profit_df = pd.DataFrame(profit_list)
                profit_df.set_index('date_time', inplace=True)

                assert cum_position_df.equals(porfolio_df[f'{direction}_position'].set_index('date_time')), \
                    f'{Fore.RED}cum_position_df is not equal to porfolio_df[f"{direction}_position"]'
                assert profit_df.equals(_event_df['profit'].set_index('date_time')), \
                    f'{Fore.RED}profit_df is not equal to _event_df["profit"]'
            # -------- debug --------
            
            porfolio_df.loc[porfolio_df[f'{direction}_position'] == 0, f'{direction}_avg_price'] = 0.0
            porfolio_df[f'{direction}_avg_price'] = porfolio_df[f'{direction}_avg_price'].fillna(method='ffill')
            if direction == 'long':
                porfolio_df[f'{direction}_flow_profit'] = porfolio_df[f'{direction}_position'] * (porfolio_df[f'{direction}_avg_price'] - self.bar_df['price'])
            else:
                porfolio_df[f'{direction}_flow_profit'] = porfolio_df[f'{direction}_position'] * (self.bar_df['price'] - porfolio_df[f'{direction}_avg_price'])
            
            porfolio_df[f'{direction}_fix_profit'] = 0.0
            porfolio_df[f'{direction}_fix_profit'] = _event_df['profit']

            porfolio_df[f'{direction}_cum_profit'] = porfolio_df[f'{direction}_flow_profit'] + porfolio_df[f'{direction}_fix_profit'].cumsum()

            porfolio_df[f'{direction}_cum_cost'] = _event_df['order_trade_cost']
            porfolio_df[f'{direction}_cum_cost'] = porfolio_df[f'{direction}_cum_cost'].cumsum()

            porfolio_df[f'{direction}_cum_capital'] = porfolio_df[f'{direction}_cum_profit'] + porfolio_df[f'{direction}_cum_cost']

            porfolio_df[f'{direction}_capital_utilization'] = 0.0
            porfolio_df[f'{direction}_capital_utilization'] = _event_df['order_asset']
            porfolio_df[f'{direction}_capital_utilization'] = porfolio_df[f'{direction}_capital_utilization'].cumsum() / self.asset_inital_value

        porfolio_df['net_position'] = porfolio_df['long_position'] - porfolio_df['short_position']
        porfolio_df['position'] = porfolio_df['long_position'] + porfolio_df['short_position']
        porfolio_df['cum_capital'] = porfolio_df['long_cum_capital'] + porfolio_df['short_cum_capital']
        porfolio_df['capital_utilization'] = porfolio_df['long_capital_utilization'] + porfolio_df['short_capital_utilization']

        porfolio_df['fix_profit'] = self.event_df['profit']
        porfolio_df['fix_profit'] = porfolio_df['fix_profit'].fillna(0.0)
        
        # ------ statistic combination ------
        pnlp_describe_df = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        pnlp_point_df = pd.DataFrame(index=['pnl_ratio', 'sharp_ratio', 'holding_ratio'])
        uw_describe_df = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        
        uw_info_tmp = FuturePerformanceStat.under_water_stat(
            'net', porfolio_df['cum_capital'], porfolio_df['position'], uw_describe_df
        )
        pnl_info_tmp = FuturePerformanceStat.pnl_position_stat(
            'net', porfolio_df['fix_profit'], porfolio_df['position'], pnlp_describe_df, pnlp_point_df
        )

        uw_long_info_tmp = FuturePerformanceStat.under_water_stat(
            'long', porfolio_df['long_cum_capital'], porfolio_df['long_position'], uw_describe_df
        )
        pnl_long_info_tmp = FuturePerformanceStat.pnl_position_stat(
            'long', porfolio_df['long_fix_profit'], porfolio_df['long_position'], pnlp_describe_df, pnlp_point_df
        )
        
        uw_short_info_tmp = FuturePerformanceStat.under_water_stat(
            'short', porfolio_df['short_cum_capital'], porfolio_df['short_position'], uw_describe_df
        )
        pnl_short_info_tmp = FuturePerformanceStat.pnl_position_stat(
            'short', porfolio_df['short_fix_profit'], porfolio_df['short_position'], pnlp_describe_df, pnlp_point_df
        )
        
        if with_detail:
            uw_info_tmp.to_csv(os.path.join(self.output_dir, 'under_water_stat.csv'))
            pnl_info_tmp.to_csv(os.path.join(self.output_dir, 'pnl_position_stat.csv'))
            uw_long_info_tmp.to_csv(os.path.join(self.output_dir, 'under_water_stat_long.csv'))
            pnl_long_info_tmp.to_csv(os.path.join(self.output_dir, 'pnl_position_stat_long.csv'))
            uw_short_info_tmp.to_csv(os.path.join(self.output_dir, 'under_water_stat_short.csv'))
            pnl_short_info_tmp.to_csv(os.path.join(self.output_dir, 'pnl_position_stat_short.csv'))
        
    @staticmethod
    def under_water_stat(
        name: str, cum_capital_series: pd.Series, position_series: pd.Series,
        uw_describe_df: pd.DataFrame
    ) -> pd.DataFrame:
        """please sort cum_capital_series and position_series by index when call this function"""
        assert cum_capital_series.index.equals(position_series.index), \
            f'{Fore.RED}cum_capital_series.index is not equal to position_series.index'

        position_key = position_series.name
        _df = pd.DataFrame(position_series)
        _df["peak"] = cum_capital_series.cummax()
        _df['underwater'] = cum_capital_series < _df['peak']
        _df["streak_id"] = (_df['underwater'] != _df['underwater'].shift(1)).cumsum()
        _df['underwater_rate'] = (cum_capital_series - _df['peak']) / _df['peak']
        _df = _df.iloc[1:][_df['underwater'] == True]
        
        under_water_stat = []
        for streak_id, streak in _df.groupby("streak_id"):
            duration = (streak.index[-1] - streak.index[0]).total_seconds() / 60
            position_ratio = (streak[position_key] != 0).sum() / streak.size
            under_water_stat.append({
                'streak_id': streak_id,
                'date_start': streak.index[0],
                'date_end': streak.index[-1],
                'size': streak.size,
                'position_duration/minutes': position_ratio * duration,
                'duration/minutes': duration,
                'max_underwater_rate': streak["underwater_rate"].min()
            })
        under_water_stat = pd.DataFrame(under_water_stat)
        under_water_stat.set_index('streak_id', inplace=True)
        
        uw_describe_df[f'{name}_duration/minutes'] = under_water_stat['duration/minutes'].describe()
        uw_describe_df[f'{name}_position_duration/minutes'] = under_water_stat['position_duration/minutes'].describe()
        uw_describe_df[f'{name}_max_underwater_rate'] = under_water_stat['max_underwater_rate'].describe()
        
        return under_water_stat

    @staticmethod
    def sharp_ratio_stat(return_series: pd.Series) -> float:
        total_time: pd.Timedelta = return_series.index[-1] - return_series.index[0]
        annualized_time = total_time.total_seconds() / 60 / 60 / 24 / 252
        return return_series.mean() / return_series.std() * np.sqrt(annualized_time)

    @staticmethod
    def pnl_position_stat(
        name: str, return_series: pd.Series, position_series: pd.Series,
        pnlp_describe_df: pd.DataFrame, pnlp_point_df: pd.DataFrame
    ) -> pd.DataFrame:
        # ------ return analysis ------
        positive_return = return_series[return_series > 0]
        negative_return = return_series[return_series < 0]
        
        # ------ position analysis ------
        _df = pd.DataFrame(index=position_series.index)
        _df['has_position'] = position_series != 0
        _df['streak_id'] = (_df['has_position'] != _df['has_position'].shift(1)).cumsum()
        _df = _df[_df['has_position'] == True]
        
        streak_stat = []
        for streak_id, streak in _df.groupby('streak_id'):
            streak_stat.append({
                'streak_id': streak_id,
                'date_start': streak.index[0],
                'date_end': streak.index[-1],
                'duration/minutes': (streak.index[-1] - streak.index[0]).total_seconds() / 60
            })
        streak_stat = pd.DataFrame(streak_stat)
        streak_stat.set_index('streak_id', inplace=True)
        
        # ------ statistic combination ------
        pnlp_describe_df[f'{name}_positive_stat'] = positive_return.describe()
        pnlp_describe_df[f'{name}_negative_stat'] = negative_return.describe()
        pnlp_describe_df[f'{name}_position_stat'] = streak_stat['duration/minutes'].describe()
        
        pnlp_point_df.loc['pnl_ratio', name] = positive_return.mean() / negative_return.mean()
        pnlp_point_df.loc['sharp_ratio', name] = FuturePerformanceStat.sharp_ratio_stat(return_series)
        pnlp_point_df.loc['holding_ratio', name] = _df.size / position_series.size
                
        return streak_stat

    def visualize(self):
        pass

