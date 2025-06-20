import numpy as np
import pandas as pd


class SampleWeightTransformer(object):
    def _num_concurrent_events(price: pd.Series, t_event: pd.Series):
        """t_event: index is the event time, value is the event duration"""
        nearest_index = price.index.searchsorted(pd.DatetimeIndex([t_event.index[0], t_event.max()]))
        count = pd.Series(0, index=price.index[nearest_index[0]:nearest_index[1] + 1])
        for t_in, t_out in t_event.iteritems():
            count.loc[t_in:t_out] += 1
        return count.loc[t_event.index.min():t_event.max()]
    
    def get_average_uniqueness(t_event: pd.Series, num_conc_events: pd.Series):
        uniqueness_weight = pd.Series(index=t_event.index)
        for t_in, t_out in t_event.iteritems():
            uniqueness_weight.loc[t_in] = (1. / num_conc_events.loc[t_in:t_out]).mean()
        uniqueness_weight /= uniqueness_weight.sum()
        return uniqueness_weight

    