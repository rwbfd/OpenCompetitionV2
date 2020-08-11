# coding = 'utf-8'
import pandas as pd
import numpy as np
import category_encoders as ce
from ..general.util import remove_continuous_discrete_prefix, split_df
import copy

import multiprocessing

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

from sklearn.decomposition import PCA

cpu_count = multiprocessing.cpu_count()


class EncoderBase(object):
    def __init__(self):
        self.trans_ls = list()

    def reset(self):
        self.trans_ls = list()

    def check_var(self, df):
        for _, _, target, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")


class CategoryEncoder(EncoderBase):
    def __init__(self):
        super().__init__()

    def fit(self, df, y, targets, configurations):
        """

        :param df: the data frame to be fitted; can be different from the transformed ones.
        :param y: the y variable
        :param targets: the variables to be transformed
        :param configurations: in the form of a list of (method, parameter), where method is one of ['woe', 'one-hot','ordinal','hash'],
        and parameter is a dictionary pertained to each encoding method
        :return:
        """
        self.reset()
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
        if method == 'target':
            self._fit_target(df, y, target, parameter)

    def _fit_hash(self, df, target):
        hash_encoder = ce.HashingEncoder()
        hash_encoder.fit(df[target].map(to_str))
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_hash' for x in
                hash_encoder.get_feature_names()]
        self.trans_ls.append(('hash', name, target, hash_encoder))

    def _fit_ordinal(self, df, target):
        ordinal_encoder = ce.OrdinalEncoder()
        ordinal_encoder.fit(df[target].map(to_str))
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_ordinal' for x in
                ordinal_encoder.get_feature_names()]
        self.trans_ls.append(('ordinal', name, target, ordinal_encoder))

    def _fit_target(self, df, y, target, parameter):
        smoothing = parameter['smoothing']
        target_encoder = ce.TargetEncoder(smoothing=smoothing)
        target_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_smooth_' + str(smoothing) + '_target' for x in
                target_encoder.get_feature_names()]
        self.trans_ls.append(('target', name, target, target_encoder))

    def _fit_one_hot(self, df, target):
        one_hot_encoder = ce.OneHotEncoder()
        target_copy = df[target].copy(deep=True)
        target_copy = target_copy.map(to_str)
        one_hot_encoder.fit(target_copy)
        name = [x + "_one_hot" for x in
                one_hot_encoder.get_feature_names()]  ## I assume that the variables start with discrete
        self.trans_ls.append(('one-hot', name, target, one_hot_encoder))

    def _fit_woe(self, df, y, target):  ##
        woe_encoder = ce.woe.WOEEncoder(cols=target)
        woe_encoder.fit(df[target].map(to_str), df[y])
        name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_woe"
        self.trans_ls.append(('woe', name, target, woe_encoder))

    def transform(self, df, y=None):
        """

        :param df: The data frame to be transformed.
        :param y: The name for y variable. Only used for leave-one-out transform for WOE and Target encoder.
        :return: The transformed dataset
        """
        for _, _, target, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")
        result_df = df.copy(deep=True)
        for method, name, target, encoder in self.trans_ls:
            if method == 'woe':
                if y:
                    result_df[name] = encoder.transform(df[target], df[y])
                else:
                    result_df[name] = encoder.transform(df[target])
            if method == 'one-hot':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'target':
                if y:
                    result_df[name] = encoder.transform(df[target].map(to_str), df[y])
                else:
                    result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'hash':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'ordinal':
                result_df[name] = encoder.transform(df[target].map(to_str))
        return result_df


class DiscreteEncoder(EncoderBase):
    def __init__(self):
        super(DiscreteEncoder, self).__init__()

    def fit(self, df, targets, configurations):
        """

        :param df: the dataframe to be fitted; can be different from the transformed one;
        :param targets: the variables to be transformed
        :param configurations: in the form of a list of (method, parameter), where method is one of ['quantile', 'uniform'],
            the parameter should contain a key called 'nbins'
        and parameter is a dictionary pertained to each encoding method
        :return:
        """
        self.reset()
        for target in targets:
            for method, parameter in configurations:
                nbins = parameter['nbins']
                self._fit_one(df, target, method, nbins)

    def _fit_one(self, df, target, method, nbins):
        if method == 'uniform':
            intervals = self._get_uniform_intervals(df, target, nbins)
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_uniform_dis_encoder"

            self.trans_ls.append((target, name, intervals))
        elif method == 'quantile':
            intervals = self._get_quantile_intervals(df, target, nbins)
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_quantile_dis_encoder"
            self.trans_ls.append((target, name, intervals))
        else:
            raise Exception("Not Implemented Yet")

    def transform(self, df):
        result = df.copy(deep=True)
        for target, _, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")

        for target, name, intervals in self.trans_ls:
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


class UnaryContinuousVarEncoder(EncoderBase):
    def __init__(self):
        super(UnaryContinuousVarEncoder, self).__init__()

    def fit(self, targets, config):
        self.reset()
        for target in targets:
            for method, parameter in config:
                if method == 'power':
                    self._fit_power(target, parameter)
                if method == 'sin':
                    self._fit_sin(target)
                if method == 'cos':
                    self._fit_cos(target)
                if method == 'tan':
                    self._fit_tan(target)
                if method == 'log':
                    self._fit_log(target)
                if method == 'exp':
                    self._fit_exp(target)
                if method == 'abs':
                    self._fit_abs(target)
                if method == 'neg':
                    self._fit_neg(target)
                if method == 'inv':
                    self._fit_inv(target)
                if method == 'sqrt':
                    self._fit_sqrt(target)

    def transform(self, df):
        result = df.copy(deep=True)
        for method, new_name, target, encoder in self.trans_ls:
            print(method)
            print(new_name)
            print(target)
            print(encoder)
            result[new_name] = result[target].apply(encoder)
        return result

    def _fit_power(self, target, parameter):
        order = parameter['order']
        _power = lambda x: np.power(x, order)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_power_" + str(order)
        self.trans_ls.append(('power', new_name, target, _power))

    def _fit_sin(self, target):
        _sin = lambda x: np.sin(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_sin'
        self.trans_ls.append(('sin', new_name, target, _sin))

    def _fit_cos(self, target):
        _cos = lambda x: np.cos(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_cos'
        self.trans_ls.append(('cos', new_name, target, _cos))

    def _fit_tan(self, target):
        _tan = lambda x: np.tan(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_tan'
        self.trans_ls.append(('tan', new_name, target, _tan))

    def _fit_log(self, target):
        _log = lambda x: np.log(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_log'
        self.trans_ls.append(('log', new_name, target, _log))

    def _fit_exp(self, target):
        _exp = lambda x: np.exp(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_exp'
        self.trans_ls.append(('exp', new_name, target, _exp))

    def _fit_abs(self, target):
        _abs = lambda x: np.abs(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_abs'
        self.trans_ls.append(('abs', new_name, target, _abs))

    def _fit_neg(self, target):
        _neg = lambda x: -(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_neg'
        self.trans_ls.append(('neg', new_name, target, _neg))

    def _fit_inv(self, target):
        _inv = lambda x: np.divide(1, x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_inv'
        self.trans_ls.append(('inv', new_name, target, _inv))

    def _fit_sqrt(self, target):
        _sqrt = lambda x: np.sqrt(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_sqrt'
        self.trans_ls.append(('sqrt', new_name, target, _sqrt))


class BinaryContinuousVarEncoder(EncoderBase):
    def __init__(self):
        super(BinaryContinuousVarEncoder, self).__init__()

    def fit(self, targets_pairs, config):
        for target1, target2 in targets_pairs:
            for method in config:
                if method == 'add':
                    self._fit_add(target1, target2)
                if method == 'sub':
                    self._fit_sub(target1, target2)
                if method == 'mul':
                    self._fit_mul(target1, target2)
                if method == 'div':
                    self._fit_div(target1, target2)

    def transform(self, df):
        result = df.copy(deep=True)
        for method, new_name, target1, target2, encoder in self.trans_ls:
            result[new_name] = result.apply(lambda row: encoder(row[target1], row[target2]), axis=1)
        return result

    def _fit_add(self, target1, target2):
        _add = lambda x, y: np.add(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_add'
        self.trans_ls.append(('add', new_name, target1, target2, _add))

    def _fit_sub(self, target1, target2):
        _sub = lambda x, y: x - y
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_sub'
        self.trans_ls.append(('sub', new_name, target1, target2, _sub))

    def _fit_mul(self, target1, target2):
        _mul = lambda x, y: np.multiply(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_mul'
        self.trans_ls.append(('mul', new_name, target1, target2, _mul))

    def _fit_div(self, target1, target2):
        _div = lambda x, y: np.divide(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_div'
        self.trans_ls.append(('div', new_name, target1, target2, _div))


class BoostTreeEncoder(EncoderBase):
    def __init__(self, nthread=None):
        super(BoostTreeEncoder, self).__init__()
        if nthread:
            self.nthread = cpu_count
        else:
            self.nthread = nthread

    def fit(self, df, y, targets_list, config):
        self.reset()
        for method, parameter in config:
            if method == 'xgboost':
                self._fit_xgboost(df, y, targets_list, parameter)
            if method == 'lightgbm':
                self._fit_lightgbm(df, y, targets_list, parameter)

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

            self.trans_ls.append(('xgb', name, targets, model, pos))

    def _fit_lightgbm(self, df, y, targets_list, parameter):
        for targets in targets_list:
            parameter_copy = copy.deepcopy(parameter)
            if 'num_threads' not in parameter.keys():
                parameter_copy['num_threads'] = self.nthread
            if 'objective' not in parameter.keys():
                parameter_copy['objective'] = "multiclass"
            num_rounds = parameter['num_threads']
            pos = parameter['pos']
            dtrain = lgb.Dataset(df[targets], label=df[y])
            model = lgb.train(parameter_copy, dtrain, num_rounds)

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.trans_ls.append(('lgb', name, targets, model, pos))

    def transform(self, df):
        result = df.copy(deep=True)
        trans_results = [result]
        for method, name, targets, model, pos in self.trans_ls:
            if method == 'xgboost':
                tree_infos: pd.DataFrame = model.trees_to_dataframe()
            elif method == 'lightgbm':
                tree_infos = tree_to_dataframe_for_lightgbm(model).get()
            else:
                raise Exception("Not Implemented Yet")

            trans_results.append(self._boost_transform(result[targets], method, name, pos, tree_infos))

        return pd.concat(trans_results, axis=1)

    @staticmethod
    def _transform_byeval(df, feature_name, leaf_condition):
        for key in leaf_condition.keys():
            if eval(leaf_condition[key]):
                df[feature_name] = key
        return df

    def _boost_transform(self, df, method, name, pos, tree_infos):
        tree_ids = tree_infos["Node"].drop_duplicates().tolist().sort()
        for tree_id in tree_ids:
            tree_info = tree_infos[tree_infos["Tree"] == tree_id][
                ["Node", "Feature", "Split", "Yes", "No", "Missing"]].copy(deep=True)
            tree_info["Yes"] = tree_info["Yes"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            tree_info["No"] = tree_info["No"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            tree_info["Missing"] = tree_info["Missing"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            leaf_nodes = tree_info[tree_info["Feature"] == "Leaf"]["Node"].drop_duplicates().tolist()
            encoder_dict = {}
            for leaf_node in leaf_nodes:
                encoder_dict[leaf_node] = get_booster_leaf_condition(leaf_node, [], tree_info)

            df.fillna(None)

            df.apply(self._transform_byeval,
                     feature_name="_".join([name, method, "tree_" + tree_id, pos]), leaf_condition=encoder_dict)

        return df


class AnomalyScoreEncoder(object):
    def __init__(self, nthread=None):
        self.result_list = list()
        if nthread:
            self.nthread = cpu_count
        else:
            self.nthread = nthread

    def fit(self, df, y, targets_list, config):
        for method, parameter in config:
            if method == 'IsolationForest':
                self._fit_isolationForest(df, y, targets_list, parameter)
            if method == 'LOF':
                self._fit_LOF(df, y, targets_list, parameter)

    def transform(self, df):
        result = df.copy(deep=True)
        for method, name, targets, model in self.result_list:
            result[name + "_" + method] = model.predict(df[targets])

        return result

    def _fit_isolationForest(self, df, y, targets_list, parameter):
        for targets in targets_list:
            n_jobs = self.nthread

            model = IsolationForest(n_jobs=n_jobs)
            model.fit(X=df[targets])

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.result_list.append(('IsolationForest', name, targets, model))

    def _fit_LOF(self, df, y, targets_list, parameter):
        for targets in targets_list:
            n_jobs = self.nthread

            model = LocalOutlierFactor(n_jobs=n_jobs)
            model.fit(X=df[targets])

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.result_list.append(("LOF", name, targets, model))


class GroupbyEncoder(EncoderBase):
    def __init__(self):
        super(GroupbyEncoder, self).__init__()

    def fit(self, df, targets, groupby_op_list):
        self.reset()
        for target in targets:
            for groupby, operations in groupby_op_list:
                for operation in operations:
                    groupby_result = self._fit_one(df, target, groupby, operation)
                    name = target + '_groupby_' + '_'.join(groupby) + '_op_' + operation
                    groupby_result = groupby_result.rename(columns={target: name})
                    self.trans_ls.append((groupby, groupby_result))

    def transform(self, df):
        result = df.copy(deep=True)
        for groupby, groupby_result in self.trans_ls:
            result = result.merge(groupby_result, on=groupby, how='left')
        return result

    def _fit_one(self, df, target, groupby_vars, operation):  # TODO: Add other aggregation options, such as kurtosis
        result = df.groupby(groupby_vars, as_index=False).agg({target: operation})
        return result


class TargetMeanEncoder(object):
    """
    This is basically a duplicate.
    """
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


def get_booster_leaf_condition(leaf_node, conditions, tree_info: pd.DataFrame):
    start_node_info = tree_info[tree_info["Node"] == leaf_node]
    if start_node_info["Feature"].tolist()[0] == "Leaf":
        conditions = []

    if str(leaf_node) in tree_info["Yes"].drop_duplicates().tolist():
        father_node_info = tree_info[tree_info["Yes"] == str(leaf_node)]
        fathers_left = True
    else:
        father_node_info = tree_info[tree_info["No"] == str(leaf_node)]
        fathers_left = False

    father_node_id = father_node_info["Node"].tolist()[0]
    split_value = father_node_info["Split"].tolist()[0]
    split_feature = father_node_info["Feature"].tolist()[0]
    if fathers_left:
        add_condition = ["row['" + split_feature + "'] <= " + str(split_value)]
        if father_node_info["Yes"].tolist()[0] == father_node_info["Missing"].tolist()[0]:
            add_condition.append("isMissing(row['" + split_feature + "'])")

    else:
        add_condition = ["row['" + split_feature + "']) > " + str(split_value)]
        if father_node_info["No"].tolist()[0] == father_node_info["Missing"].tolist()[0]:
            add_condition.append("row['" + split_feature + "'] == None")

    add_condition = "(" + " or ".join(add_condition) + ")"
    conditions.append(add_condition)

    if father_node_info["Node"].tolist()[0] == 0:
        return " and ".join(conditions)
    else:
        return get_booster_leaf_condition(father_node_id, conditions, tree_info)


class tree_to_dataframe_for_lightgbm(object):
    def __init__(self, model):
        self.json_model = model.dump_model()
        self.features = self.json_model["feature_names"]

    def get_root_nodes_count(self, tree, max_id):
        tree_node_id = tree.get("split_index")
        if tree_node_id:
            if tree_node_id > max_id:
                max_id = tree_node_id

        if tree.get("left_child"):
            left = self.get_root_nodes_count(tree.get("left_child"), max_id)
            if left > max_id:
                max_id = left
        else:
            left = []

        if tree.get("right_child"):
            right = self.get_root_nodes_count(tree.get("right_child"), max_id)
            if right > max_id:
                max_id = right
        else:
            right = []

        if not left and not right:  # 如果root是叶子结点
            max_id = max_id
        return max_id

    def get(self):
        tree_dataframe = []

        for tree in self.json_model["tree_info"]:
            tree_id = tree["tree_index"]
            tree = tree["tree_structure"]
            root_nodes_count = self.get_root_nodes_count(tree, 0) + 1
            tree_dataFrame = pd.DataFrame()
            tree_df = self._lightGBM_trans(tree, tree_dataFrame, tree_id, root_nodes_count).sort_values(
                "Node").reset_index(drop=True)
            tree_df["Tree"] = tree_id
            tree_dataframe.append(tree_df)

        return pd.concat(tree_dataframe, axis=0)

    def _lightGBM_trans(self, tree, tree_dataFrame, tree_id, root_nodes_count):
        tree_node_id = tree.get("split_index")
        threshold = tree.get("threshold")
        default_left = tree.get("default_left")

        if tree_node_id is not None:
            data = {"Node": tree_node_id, "Feature": self.features[tree.get("split_index")], "Split": threshold}
            yes_id = tree.get("left_child").get("split_index")
            if yes_id is None:
                yes_id = tree.get("left_child").get("leaf_index") + root_nodes_count
            tree_dataFrame = self._lightGBM_trans(tree.get("left_child"), tree_dataFrame, tree_id, root_nodes_count)

            no_id = tree.get("right_child").get("split_index")
            if no_id is None:
                no_id = tree.get("right_child").get("leaf_index") + root_nodes_count

            tree_dataFrame = self._lightGBM_trans(tree.get("right_child"), tree_dataFrame, tree_id, root_nodes_count)

            if default_left:
                missing_id = yes_id
            else:
                missing_id = no_id
            data["Yes"], data["No"], data["Missing"] = "_".join([tree_id, yes_id]), "_".join(
                [tree_id, no_id]), "_".join([tree_id, missing_id])
        else:
            # print(tree)
            # print(tree_node_id)
            data = {"Node": root_nodes_count + tree.get("leaf_index"), "Feature": "Leaf", "Split": None, "Yes": None,
                    "No": None, "Missing": None}

        row = pd.DataFrame.from_dict(data, orient="index").T
        tree_dataFrame = pd.concat([tree_dataFrame, row])
        return tree_dataFrame


class StandardizeEncoder(EncoderBase):
    def __init__(self):
        super(StandardizeEncoder, self).__init__()

    def fit(self, df, targets, ):
        self.reset()
        for target in targets:
            mean = df[target].mean()
            std = df[target].std()
            new_name = 'continuous_standardized_' + remove_continuous_discrete_prefix(target)
            self.trans_ls.append((target, mean, std, new_name))

    def transform(self, df):
        result = df.copy(deep=True)
        for target, mean, std, new_name in self.trans_ls:
            result[new_name] = (result[target] - mean) / std
        return result


class InteractionEncoder:
    def __init__(self):
        self.level = list()
        self.targets = None

    def fit(self, targets, level='all'):
        if level == 'all':
            self.level = [2, 3, 4]
        else:
            self.level = level
        self.targets = targets

    def transform(self, df):
        result = df.copy(deep=True)
        for level in self.level:
            if level == 2:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        new_name = 'continuous_' + remove_continuous_discrete_prefix(
                            target_1) + "_" + remove_continuous_discrete_prefix(target_2) + "_cross"
                        result[new_name] = result[target_1] * result[target_2]
            if level == 3:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        for target_3 in self.targets:
                            new_name = 'continuous_' + remove_continuous_discrete_prefix(
                                target_1) + "_" + remove_continuous_discrete_prefix(
                                target_2) + "_" + remove_continuous_discrete_prefix(target_3) + "_cross"

                            result[new_name] = result[target_1] * result[target_2] * result[target_3]
            if level == 4:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        for target_3 in self.targets:
                            for target_4 in self.targets:
                                new_name = 'continuous_' + remove_continuous_discrete_prefix(
                                    target_1) + "_" + remove_continuous_discrete_prefix(
                                    target_2) + "_" + remove_continuous_discrete_prefix(
                                    target_3) + "_" + remove_continuous_discrete_prefix(target_4) + "_cross"
                                result[new_name] = result[target_1] * result[target_2] * result[target_3] * result[
                                    target_4]
        return result


class DimReducEncoder:
    def __init__(self):
        self.result = list()

    def fit(self, df, targets, config):
        for target in targets:
            for method, parameter in config:
                if method == 'pca':
                    n_comp = config['n_components']
                    pos = config['pos']
                    encoder = PCA(n_comp)
                    PCA.fit(df[target])
                    self.result.append((method, encoder, pos, n_comp, target))

    def transform(self, df):
        result = df.copy(deep=True)
        for method, encoder, pos, n_comp, target in self.result:
            if method == 'pca':
                new_names = ["_pca_" + str(x) + "_pos" for x in range(n_comp)]
                result[new_names] = encoder.transform(df[target])
        return result
