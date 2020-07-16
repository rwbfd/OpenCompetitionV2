# coding = 'utf-8'
import pandas as pd
import numpy as np
import category_encoders as ce
from ..general.util import remove_continuous_discrete_prefix, split_df
import copy
import multiprocessing

cpu_count = multiprocessing.cpu_count()

import xgboost as xgb


class CategoryEncoder(object):  # TODO: For each of them, need to add possibility for multivariate classification
    def __init__(self):
        self.result_list = list()

    def fit(self, df, y, targets, configurations):
        for target in targets:
            for config in configurations:
                self._fit_one(df, y, target, config)

    def _fit_one(self, df, y, target, config):
        method, parameter = config[0], config[1]
        if method == 'woe':
            self._fit_woe(df, y, target)
        if method == 'one-hot':
            self._fit_one_hot(df, target)
        if method == 'ordinal':
            self._fit_ordinal(df, target)
        if method == 'hash':
            self._fit_hash(df, target)

    def _fit_hash(self, df, target):
        hash_encoder = ce.HashingEncoder()
        hash_encoder.fit(df[target])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_hash' for x in
                hash_encoder.get_feature_names()]
        self.result_list.append(('hash', name, target, hash_encoder))

    def _fit_ordinal(self, df, target):
        ordinal_encoder = ce.OrdinalEncoder()
        ordinal_encoder.fit(df[target])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_ordinal' for x in
                ordinal_encoder.get_feature_names()]
        self.result_list.append(('ordinal', name, target, ordinal_encoder))

    def _fit_target(self, df, y, target, parameter):
        smoothing = parameter['smoothing']
        target_encoder = ce.TargetEncoder(smoothing=smoothing)
        target_encoder.fit(df[target], df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_smooth_' + str(smoothing) + '_target' for x in
                target_encoder.get_feature_names()]
        self.result_list.append(('target', name, target, target_encoder))

    def _fit_one_hot(self, df, target):
        one_hot_encoder = ce.OneHotEncoder()
        target_copy = df[target].copy(deep=True)
        target_copy = target_copy.map(to_str)
        one_hot_encoder.fit(target_copy)
        name = [x + "_one_hot" for x in
                one_hot_encoder.get_feature_names()]  ## I assume that the variables start with discrete
        self.result_list.append(('one-hot', name, target, one_hot_encoder))

    def _fit_woe(self, df, y, target):  ##
        woe_encoder = ce.woe.WOEEncoder(cols=target)
        woe_encoder.fit(df[target], df[y])
        name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_woe"
        self.result_list.append(('woe', name, target, woe_encoder))

    def transform(self, df, y=None):  ### TODO: This can be optimized
        result_df = df.copy(deep=True)
        for method, name, target, encoder in self.result_list:
            if method == 'woe':
                if y:
                    result_df[name] = encoder.transform(df[target], df[y])
                else:
                    result_df[name] = encoder.transform(df[target])
            if method == 'one-hot':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'target':
                result_df[name] = encoder.transform(df[target])
            if method == 'hash':
                result_df[name] = encoder.transform(df[target])
            if method == 'ordinal':
                result_df[name] = encoder.transform(df[target])
        return result_df


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
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_uniform_dis_encoder"

            self.result_list.append((target, name, intervals))
        elif method == 'quantile':
            intervals = self._get_quantile_intervals(df, target, nbins)
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_quantile_dis_encoder"
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


class BoostTreeEncoder:  ## TODO: Please add LightGBM and CatBoost
    def __init__(self, nthread=None):
        self.result_list = list()
        if nthread:
            self.nthread = cpu_count
        else:
            self.nthread = nthread

    def fit(self, df, y, targets_list, config):
        for method, parameter in config:
            if method == 'xgboost':
                self._fit_xgboost(df, y, targets_list, parameter)

    def transform(self, df, targets):  ## TODO: It doesn't seem we need target here, maybe we don't need others as well?
        pass

    def _fit_xgboost(self, df, y, targets_list, parameter):
        for targets in targets_list:

            parameter_copy = copy.deepcopy(parameter)
            if 'nthread' not in parameter.keys():
                parameter_copy['nthread'] = self.nthread
            if 'objective' not in parameter.keys():
                parameter_copy['objective'] = "multi:softmax"
            num_rounds = parameter['num_rounds']
            pos = parameter['pos']
            dtrain = xgb.DMatrix(df[targets], label=df[y])
            model = xgb.train(parameter_copy, dtrain, num_rounds)
            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)

            self.result_list.append(('xgb', name, targets, model, pos))


class GroupbyEncoder(object):
    def __init__(self):
        self.groupby_result_list = list()

    def fit(self, df, targets, groupby_op_list):
        self.groupby_result_list = list()
        for target in targets:
            for groupby, operations in groupby_op_list:
                for operation in operations:
                    groupby_result = self._fit_one(df, target, groupby, operation)
                    name = target + '_groupby_' + '_'.join(groupby) + '_op_' + operation
                    groupby_result = groupby_result.rename(columns={target: name})
                    self.groupby_result_list.append((groupby, groupby_result))

    def transform(self, df):
        result = df.copy(deep=True)
        for groupby, groupby_result in self.groupby_result_list:
            result = result.merge(groupby_result, on=groupby, how='left')
        return result

    def _fit_one(self, df, target, groupby_vars, operation):  # TODO: Add other aggregation options, such as kurtosis
        result = df.groupby(groupby_vars, as_index=False).agg({target: operation})
        return result


class TargetMeanEncoder(object):
    def __init__(self, smoothing_coefficients=None):
        if not smoothing_coefficients:
            self.smoothing_coefficients = [1]
        else:
            self.smoothing_coefficients = smoothing_coefficients

    def fit_and_transform_train(self, df_train, ys, target_vars, n_splits=5):
        splitted_df = split_df(df_train, n_splits=n_splits, shuffle=True)
        result = list()
        for train_df, test_df in splitted_df:
            for y in ys:
                for target_var in target_vars:
                    for smoothing_coefficient in self.smoothing_coefficients:
                        test_df = self._fit_one(train_df, test_df, y, target_var, smoothing_coefficient)
            result.append(test_df)
        return pd.concat(result)

    def _fit_one(self, train_df, test_df, y, target_var, smoothing_coefficient):
        global_average = train_df[y].mean()
        local_average = train_df.groupby(target_var)[y].mean().to_frame().reset_index()
        name = "target_mean_" + y + "_" + target_var + "_lambda_" + str(smoothing_coefficient)
        local_average = local_average.rename(columns={y: name})
        test_df = test_df.merge(local_average, on=target_var, how='left')
        test_df[name] = test_df[name].map(
            lambda x: global_average if pd.isnull(x) else smoothing_coefficient * x + (
                    1 - smoothing_coefficient) * global_average)
        return test_df


class UnaryContinuousVarEncoder:
    def __init__(self):
        self.result_list = list()

    def fit(self, df, y, targets, config):
        for target in targets:
            for method, parameter in config:
                if method == 'power':
                    pass
                if method == 'sin':
                    pass
                if method == 'cos':
                    pass
                if method == 'bc':
                    pass
                if method == 'jy':
                    pass


class BinaryContinuousVarEncoder:
    def __init__(self):
        self.result_list = list()

    def fit(self, df, y, targets_pairs, config):
        pass


def get_interval(x, sorted_intervals):  ### Needs to be rewritten to remove found and duplicated code
    interval = 0
    found = False

    if pd.isnull(x):
        return np.nan
    if x < sorted_intervals[0] or x > sorted_intervals[-1]:
        return np.nan
    while not found and interval < len(sorted_intervals) - 1:
        if sorted_intervals[interval] <= x <= sorted_intervals[interval + 1]:
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


def to_str(x):
    if pd.isnull(x):
        return '#NA#'
    else:
        return str(x)
