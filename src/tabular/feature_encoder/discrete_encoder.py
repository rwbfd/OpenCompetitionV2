# coding = 'utf-8'
import pandas as pd
import numpy as np


class DiscreteEncoder(object):
    def __init__(self):
        self.result_list = list()

    def fit(self, df, targets, configurations):
        self.result_list = list()
        for target in targets:
            for method, nbins in configurations:
                self._fit_one(df, target, method, nbins)

    def _fit_one(self, df, target, method, nbins):
        if method == 'uniform':
            intervals = self._get_uniform_intervals(df, target, nbins)
            name = target + "_uniform_" + str(nbins)
            self.result_list.append((target, name, intervals))
        elif method == 'quantile':
            intervals = self._get_quantile_intervals(df, target, nbins)
            name = target + "_quantile_" + str(nbins)
            self.result_list.append((target, name, intervals))
        else:
            raise Exception("Not Implemented Yet")

    def transform(self, df):
        result = df.copy(deep=True)
        for target, name, intervals in self.result_list:
            result[name] = result[target].map(lambda x: get_interval(x, intervals))
        return result

    def _get_uniform_intervals(self, df, target, nbins):
        target_var = df[target]
        minimum = target_var.min()
        maximum = target_var.max()

        intervals = get_uniform_interval(minimum, maximum, nbins)
        return intervals

    def _get_quantile_intervals(self, df, target, nbins):
        return get_quantile_interval(df[target], nbins)


def get_interval(x, sorted_intervals):
    interval = 0
    found = False

    if pd.isnull(x):
        return np.nan
    if x < sorted_intervals[0] or x > sorted_intervals[-1]:
        return np.nan
    while not found and interval < len(sorted_intervals) - 1:
        if sorted_intervals[interval] <= x <= sorted_intervals[interval + 1]:
            found = True
            return "i_" + str(interval)
        else:
            interval += 1



def get_uniform_interval(minimum, maximum, nbins):
    result = [minimum]
    step_size = (float(maximum - minimum)) / nbins
    for index in range(nbins - 1):
        result.append(minimum + step_size * (index + 1))
    result.append(maximum)
    return result


def get_quantile_interval(data, nbins):
    quantiles = get_uniform_interval(0, 1, nbins)
    return list(data.quantile(quantiles))