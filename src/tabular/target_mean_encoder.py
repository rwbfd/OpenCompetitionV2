# coding = 'utf-8'
from ..general.data_util import split_df
import pandas as pd


class TargetMeanEncoder(object):
    def __init__(self, smoothing_coefficients=None):
        if not smoothing_coefficients:
            self.smoothing_coefficients = [1]
        else:
            self.smoothing_coefficients = smoothing_coefficients

    def encode(self, df, ys, target_vars, n_splits=5):
        splitted_df = split_df(df, n_splits=n_splits, shuffle=True)
        result = list()
        for train_df, test_df in splitted_df:
            for y in ys:
                for target_var in target_vars:
                    for smoothing_coefficient in self.smoothing_coefficients:
                        test_df = self._encode_one(train_df, test_df, y, target_var, smoothing_coefficient)
            result.append(test_df)
        return pd.concat(result)

    def _encode_one(self, train_df, test_df, y, target_var, smoothing_coefficient):
        global_average = train_df[y].mean()
        local_average = train_df.groupby(target_var)[y].mean().to_frame().reset_index()
        name = "target_mean_" + y + "_" + target_var + "_lambda_" + str(smoothing_coefficient)
        local_average = local_average.rename(columns={y: name})
        test_df = test_df.merge(local_average, on=target_var, how='left')
        test_df[name] = test_df[name].map(
            lambda x: global_average if pd.isnull(x) else smoothing_coefficient * x + (
                    1 - smoothing_coefficient) * global_average)
        return test_df
