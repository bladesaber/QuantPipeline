import pandas as pd
import numpy as np


class TimeBarsResampler(object):
    def __init__(self, num_units: int, resolution: str):
        self.bar_list = []
        self.time_bar_thresh_mapping = {'D': 86400, 'H': 3600, 'MIN': 60, 'S': 1}
        self.num_units = num_units
        self.resolution = resolution
        self.threshold = self.num_units * self.time_bar_thresh_mapping[self.resolution]
        
        self.prev_dict = {
            'price': None, 'signal': None,
        }
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
        }
        self.is_first_bar = True
        
    def _reset_cache(self):
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
            'open': None, 'close': None,
        }
        self.is_first_bar = True
    
    def _extract_bar(self, data: pd.DataFrame):
        self.bar_list.clear()
        timestamp_threshold = int(data.iloc[0].datetime.timestamp() // self.threshold + 1) * self.threshold
        for i, row in enumerate(data.itertuples()):
            date_time = row.datetime.timestamp()
            
            if i == 0:
                self.prev_dict['price'] = row.price
                signal = 0
            else:
                signal = np.sign(row.price - self.prev_dict['price'])
                if signal == 0:
                    signal = self.prev_dict['signal']
                self.prev_dict['price'] = row.price
                self.prev_dict['signal'] = signal
            
            self.cum_statistics['ticks_num'] += 1
            self.cum_statistics['dollar_value'] += row.price * row.volume
            self.cum_statistics['volume'] += row.volume
            self.cum_statistics['high'] = max(self.cum_statistics['high'], row.high)
            self.cum_statistics['low'] = min(self.cum_statistics['low'], row.low)
            if signal > 0:
                self.cum_statistics['long_ticks_num'] += 1
                self.cum_statistics['long_dollar_value'] += row.price * row.volume
                self.cum_statistics['long_volume'] += row.volume
            elif signal < 0:
                self.cum_statistics['short_ticks_num'] += 1
                self.cum_statistics['short_dollar_value'] += row.price * row.volume
                self.cum_statistics['short_volume'] += row.volume
            
            if self.is_first_bar:
                self.prev_dict['open'] = row.price
                self.is_first_bar = False
                self.cum_statistics['start_timestamp'] = row.datetime
            if date_time >= timestamp_threshold:
                # ------ update the timestamp threshold ------
                timestamp_threshold = int(date_time // self.threshold + 1) * self.threshold
                
                # ------ create the bar ------
                self.cum_statistics['close'] = row.price
                self.cum_statistics['end_timestamp'] = row.datetime
                self.bar_list.append(self.cum_statistics.copy())
                self._reset_cache()
                
        return self.bar_list


class ConstBarsResampler(TimeBarsResampler):
    def __init__(self, metric: str, metric_threshold: float):
        self.bar_list = []
        self.metric = metric
        self.metric_threshold = metric_threshold
        
        self.prev_dict = {'price': None, 'signal': None}
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
        }
        self.is_first_bar = True
        self.threshold_dict = {'cum_imbalance': 0}
    
    def _reset_cache(self):
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
        }
        self.is_first_bar = True
        self.threshold_dict.update({'cum_imbalance': 0})
        
    def _extract_bar(self, data: pd.DataFrame):
        self.bar_list.clear()
        for i, row in enumerate(data.itertuples()):            
            if i == 0:
                self.prev_dict['price'] = row.price
                signal = 0
            else:
                signal = np.sign(row.price - self.prev_dict['price'])
                if signal == 0:
                    signal = self.prev_dict['signal']
                self.prev_dict['price'] = row.price
                self.prev_dict['signal'] = signal
            
            imbalance = self._get_metric(row.price, row.volume, signal)
            self.threshold_dict['cum_imbalance'] += imbalance
            
            self.cum_statistics['ticks_num'] += 1
            self.cum_statistics['dollar_value'] += row.price * row.volume
            self.cum_statistics['volume'] += row.volume
            self.cum_statistics['high'] = max(self.cum_statistics['high'], row.high)
            self.cum_statistics['low'] = min(self.cum_statistics['low'], row.low)
            if signal > 0:
                self.cum_statistics['long_ticks_num'] += 1
                self.cum_statistics['long_dollar_value'] += row.price * row.volume
                self.cum_statistics['long_volume'] += row.volume
            elif signal < 0:
                self.cum_statistics['short_ticks_num'] += 1
                self.cum_statistics['short_dollar_value'] += row.price * row.volume
                self.cum_statistics['short_volume'] += row.volume
            
            if self.is_first_bar:
                self.prev_dict['open'] = row.price
                self.is_first_bar = False
                self.cum_statistics['start_timestamp'] = row.datetime
                
            if self.threshold_dict['cum_imbalance'] >= self.metric_threshold and i > 0:
                self.cum_statistics['close'] = row.price
                self.cum_statistics['end_timestamp'] = row.datetime
                self.bar_list.append(self.cum_statistics.copy())
                self._reset_cache()

    def _get_metric(self, price: float, volume: float, signal: float):
        if self.metric == 'signal_imbalance':
            imbalance = signal
        elif self.metric == 'signal_volume_imbalance':
            imbalance = volume * signal
        elif self.metric == 'volume_imbalance':
            imbalance = volume
        elif self.metric == 'dollar_value_imbalance':
            imbalance = price * volume
        elif self.metric == 'signal_dollar_value_imbalance':
            imbalance = signal * price * volume
        elif self.metric == 'ticks':
            imbalance = 1
        else:
            raise ValueError(f"Invalid metric: {self.metric}")
        return imbalance


class EmaBarsResampler(ConstBarsResampler):
    def __init__(
        self, metric: str, expected_ticks_num_initial: int, expected_imbalance_initial: float,
        ewma_weight: float = 0.8, ewma_size: int = 5
    ):
        self.bar_list = []
        self.metric = metric
        self.ewma_size = ewma_size
        self.ewma_weight = EmaBarsResampler._get_ewma_weight(ewma_weight, ewma_size)
        
        self.prev_dict = {'price': None, 'signal': None, 'tick_idx': 0}
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
        }
        self.is_first_bar = True
        self.threshold_dict = {
            'cum_imbalance': 0, 
            'imbalance_array': [],
            'ticks_num_array': [],
            'expected_imbalance': expected_imbalance_initial,
            'expected_ticks_num': expected_ticks_num_initial,
        }
        
    def _reset_cache(self):
        self.cum_statistics = {
            'start_timestamp': None, 'end_timestamp': None,
            'ticks_num': 0, 'long_ticks_num': 0,
            'dollar_value': 0, 'long_dollar_value': 0,
            'volume': 0, 'long_volume': 0,
            'high': -np.inf, 'low': np.inf,
        }
        self.is_first_bar = True
        self.threshold_dict.update({
            'cum_imbalance': 0
        })
    
    def _extract_bar(self, data: pd.DataFrame):
        self.bar_list.clear()
        for i, row in enumerate(data.itertuples()):
            if i == 0:
                self.prev_dict['price'] = row.price
                signal = 0
            else:
                signal = np.sign(row.price - self.prev_dict['price'])
                if signal == 0:
                    signal = self.prev_dict['signal']
                self.prev_dict['price'] = row.price
                self.prev_dict['signal'] = signal
            
            imbalance = self._get_metric(row.price, row.volume, signal)
            self.threshold_dict['cum_imbalance'] += imbalance
            self.threshold_dict['imbalance_array'].append(imbalance)
            
            self.cum_statistics['ticks_num'] += 1
            self.cum_statistics['dollar_value'] += row.price * row.volume
            self.cum_statistics['volume'] += row.volume
            self.cum_statistics['high'] = max(self.cum_statistics['high'], row.high)
            self.cum_statistics['low'] = min(self.cum_statistics['low'], row.low)
            if signal > 0:
                self.cum_statistics['long_ticks_num'] += 1
                self.cum_statistics['long_dollar_value'] += row.price * row.volume
                self.cum_statistics['long_volume'] += row.volume
            elif signal < 0:
                self.cum_statistics['short_ticks_num'] += 1
                self.cum_statistics['short_dollar_value'] += row.price * row.volume
                self.cum_statistics['short_volume'] += row.volume
            
            if self.is_first_bar:
                self.prev_dict['open'] = row.price
                self.is_first_bar = False
                self.cum_statistics['start_timestamp'] = row.datetime
                
            if self.threshold_dict['cum_imbalance'] >= self.threshold_dict['expected_imbalance'] * self.threshold_dict['expected_ticks_num']:
                # ------ calculate the tick number of the current bar ------
                tick_num = i - self.prev_dict['tick_idx']
                self.threshold_dict['ticks_num_array'].append(tick_num)
                self.prev_dict['tick_idx'] = i
                
                # ------ create the bar ------
                self.cum_statistics['close'] = row.price
                self.cum_statistics['end_timestamp'] = row.datetime
                self.bar_list.append(self.cum_statistics.copy())
                self._reset_cache()
                
                # ------ update the expected imbalance and ticks number ------
                self._update_expected_imbalance()
                self._update_expected_ticks_num()
    
    def _update_expected_imbalance(self):
        if len(self.threshold_dict['imbalance_array']) >= self.ewma_size:
            self.threshold_dict['expected_imbalance'] = np.sum(self.ewma_weight * \
                self.threshold_dict['imbalance_array'][-self.ewma_size:])
    
    def _update_expected_ticks_num(self):
        if len(self.threshold_dict['ticks_num_array']) >= self.ewma_size:
            self.threshold_dict['expected_ticks_num'] = np.sum(self.ewma_weight * \
                self.threshold_dict['ticks_num_array'][-self.ewma_size:])
    
    @staticmethod
    def _get_ewma_weight(weight: float, size: int):
        weights = np.array([weight * (1 - weight) ** i for i in range(size)])[::-1]
        return weights / weights.sum()
                
