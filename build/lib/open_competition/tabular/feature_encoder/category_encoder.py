# coding = 'utf-8'

import category_encoders as ce
import pandas as pd


class CategoryEncoder(object):
    def __init__(self):
        self.result_list = list()

    def fit(self, df, targets, configurations):
        self.result_list = list()
        for target in targets:
            for method in configurations:
                self._fit_one(df, target, method)

    def _fit_one(self, df, target_col, method):
        if method == 'ordinal':
            encoder, intervals = self._get_ordinal_intervals(df, target_col)
            name = target_col + "_ordinal"
            self.result_list.append((target_col, [name], encoder))

        elif method == 'one-hot':
            encoder, intervals = self._get_one_hot_intervals(df, target_col)
            name = [item.replace("_", "_onehot_") for item in list(intervals.columns)]
            self.result_list.append((target_col, name, encoder))

        elif method == 'hash':
            encoder, intervals = self._get_hash_intervals(df, target_col)
            name = [item.replace("col", target_col + "_onehot_") for item in list(intervals.columns)]
            self.result_list.append((target_col, name, encoder))

        else:
            raise Exception("Not Implemented Yet")

    def _get_ordinal_intervals(self, df, target_col):
        target_var = df[target_col]
        encoder = ce.ordinal.OrdinalEncoder(cols=target_col)
        encoder.fit(target_var)
        return encoder, encoder.transform(target_var)

    def _get_one_hot_intervals(self, df, target_col):
        target_var = df[target_col]
        encoder = ce.one_hot.OneHotEncoder(cols=target_col)
        encoder.fit(target_var)
        return encoder, encoder.transform(target_var)

    def _get_hash_intervals(self, df, target_col):
        target_var = df[target_col]
        encoder = ce.hashing.HashingEncoder(cols=target_col)
        encoder.fit(target_var)
        return encoder, encoder.transform(target_var)

    def transform(self, df):
        results = df.copy(deep=True)
        res_list = list()

        for target_col, name, encoder in self.result_list:
            res = encoder.transform(results[target_col])
            res.columns = name
            res_list.append(res)
        results = pd.concat([results] + res_list, axis=1)
        return results
