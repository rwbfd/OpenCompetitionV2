# coding = 'utf-8'
from copy import deepcopy
import itertools
import math
from dataclasses import dataclass, asdict
import multiprocessing
from hyperopt import fmin, tpe, hp
import hyperopt.pyll
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import io
from contextlib import redirect_stdout

cpu_count = multiprocessing.cpu_count()


@dataclass
class XGBOpt:
    nthread: hyperopt.pyll.base.Apply = hp.choice('nthread', [cpu_count])
    eval_metric: hyperopt.pyll.base.Apply = hp.choice('eval_metric', ['error'])
    objective: hyperopt.pyll.base.Apply = hp.choice('objective', ['binary:logistic'])
    max_depth: hyperopt.pyll.base.Apply = hp.choice('max_depth', [4, 5, 6, 7, 8])
    early_stopping_rounds: hyperopt.pyll.base.Apply = hp.choice('early_stopping_rounds', [50])
    num_round: hyperopt.pyll.base.Apply = hp.choice('num_round', [1000])
    eta: hyperopt.pyll.base.Apply = hp.uniform('eta', 0.01, 0.011)
    subsample: hyperopt.pyll.base.Apply = hp.uniform('subsample', 0.8, 1)
    colsample_bytree: hyperopt.pyll.base.Apply = hp.uniform('colsample_bytree', 0.3, 1)
    gamma: hyperopt.pyll.base.Apply = hp.choice('gamma', [0, 1, 5])


@dataclass
class LGBOpt:
    num_threads: hyperopt.pyll.base.Apply = hp.choice('num_threads', [cpu_count])
    num_leaves: hyperopt.pyll.base.Apply = hp.choice('num_leaves', [64])
    metric: hyperopt.pyll.base.Apply = hp.choice('metric', ['binary_error'])
    num_round: hyperopt.pyll.base.Apply = hp.choice('num_rounds', [1000])
    objective: hyperopt.pyll.base.Apply = hp.choice('objective', ['binary'])
    learning_rate: hyperopt.pyll.base.Apply = hp.uniform('learning_rate', 0.01, 0.1)
    feature_fraction: hyperopt.pyll.base.Apply = hp.uniform('feature_fraction', 0.5, 1.0)
    bagging_fraction: hyperopt.pyll.base.Apply = hp.uniform('bagging_fraction', 0.8, 1.0)


class FitterBase(object):
    def __init__(self, label, metric, max_eval=100, opt=None):
        self.label = label
        self.metric = metric
        self.opt_params = dict()
        self.max_eval = max_eval
        self.opt = opt

    def get_loss(self, y, y_pred):
        if self.metric == 'error':
            return 1 - accuracy_score(y, y_pred)
        elif self.metric == 'precision':
            return 1 - precision_score(y, y_pred)
        elif self.metric == 'recall':
            return 1 - recall_score(y, y_pred)
        elif self.metric == 'macro_f1':
            return 1 - f1_score(y, y_pred, average='macro')
        elif self.metric == 'micro_f1':
            return 1 - f1_score(y, y_pred, average='micro')
        elif self.metric == 'auc':  # TODO: Add a warning checking if y_predict is all [0, 1], it should be probability
            return 1 - roc_auc_score(y, y_pred)
        else:
            raise Exception("Not implemented yet.")

    def get_rand_param(self):
        if self.opt is None:
            return None
        else:
            return hyperopt.pyll.stochastic.sample(asdict(self.opt))


class XgBoostFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: XGBOpt = None, max_eval=100):
        super(XgBoostFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = XGBOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None, use_early_stop=True, verbose_eval=False):
        dtrain = xgb.DMatrix(train_df.drop(columns=[self.label]), train_df[self.label])
        deval = xgb.DMatrix(eval_df.drop(columns=[self.label]), eval_df[self.label])
        evallist = [(deval, 'eval')]
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        if use_early_stop:
            self.clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                 early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=verbose_eval)
        else:
            self.clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                 verbose_eval=verbose_eval)

    def search(self, train_df, eval_df, use_early_stop=True, verbose_eval=False):
        self.opt_params = dict()
        deval = xgb.DMatrix(eval_df.drop(columns=[self.label]))

        def train_impl(params):
            self.train(train_df, eval_df, params, use_early_stop=use_early_stop, verbose_eval=verbose_eval)
            if self.metric == 'auc':
                y_pred = self.clf.predict(deval)
            else:
                y_pred = (self.clf.predict(deval) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data, use_early_stop=True, verbose_eval=False):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.spilt(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params, use_early_stop=use_early_stop, verbose_eval=verbose_eval)
                deval = xgb.DMatrix(eval_df.drop(columns=[self.label]))
                if self.metric == 'auc':
                    y_pred = self.clf.predict(deval)
                else:
                    y_pred = (self.clf.predict(deval) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, use_early_stop=True,
                     verbose_eval=False, drop_test_y=True):
        if params is not None:
            use_params = deepcopy(params)
        else:
            use_params = deepcopy(self.opt_params)
        train_pred = np.ndarray([np.NaN for x in range(train_data.shape[0])])
        test_pred = np.ndarray([0 for x in range(test_data.shape[0])])
        if drop_test_y:
            dtest = xgb.DMatrix(test_data.drop(columns=self.label))
        else:
            dtest = xgb.DMatrix(test_data)
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            dtrain = xgb.DMatrix(train_df.drop(columns=[self.label]), train_df[self.label])
            deval = xgb.DMatrix(eval_df.drop(columns=[self.label]), eval_df[self.label])
            evallist = [(deval, 'eval')]
            if use_early_stop:
                clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'],
                                early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=verbose_eval)
            else:
                clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                verbose_eval=verbose_eval)
            train_pred[eval_id] = clf.predict(deval)
            test_pred += clf.predict(dtest)
        test_pred = test_pred / k_fold.n_splits
        return train_pred, test_pred


class LGBFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: LGBOpt = None, max_eval=100):
        super(LGBFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = LGBOpt()

    def train(self, train_df, eval_df, params=None, use_best_eval=True):
        dtrain = lgb.Dataset(train_df.drop(columns=[self.label]), train_df[self.label])
        deval = lgb.Dataset(eval_df.drop(columns=[self.label]), eval_df[self.label])
        evallist = [dtrain, deval]
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)

        num_round = use_params.pop('num_round')
        if use_best_eval:
            with io.StringIO() as buf, redirect_stdout(buf):
                clf = lgb.train(use_params, dtrain, num_round, valid_sets=evallist)
                output = buf.getvalue().split("\n")
            min_error = np.inf
            min_index = 0
            for idx in range(len(output) - 1):
                if min_error > float(output[idx].split("\t")[2].split(":")[1]):
                    min_error = float(output[idx].split("\t")[2].split(":")[1])
                    min_index = idx
            print("The minimum is attained in round %d" % (min_index + 1))
            self.clf = lgb.train(use_params, dtrain, min_index + 1, valid_sets=evallist,
                                 verbose_eval=False)
            return output
        else:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = lgb.train(use_params, dtrain, num_round, valid_sets=evallist)
                output = buf.getvalue().split("\n")
            return output

    def search(self, train_df, eval_df, use_best_eval=True):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params, use_best_eval)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = (self.clf.predict(eval_df.drop(columns=[self.label])) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data, use_best_eval=True):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params, use_best_eval)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = (self.clf.predict(eval_df.drop(columns=[self.label])) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True, use_best_eval=True):
        acc_result = list()
        if params is not None:
            use_params = params
        else:
            use_params = self.opt_params
        num_round = use_params.pop('num_round')
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            dtrain = lgb.Dataset(train_df.drop(columns=[self.label]), train_df[self.label])
            deval = lgb.Dataset(eval_df.drop(columns=[self.label]), eval_df[self.label])
            evallist = [dtrain, deval]
            if use_best_eval:
                with io.StringIO() as buf, redirect_stdout(buf):
                    lgb.train(use_params, dtrain, num_round, valid_sets=evallist)
                    output = buf.getvalue().split("\n")
                min_error = np.inf
                min_index = 0
                for idx in range(len(output) - 1):
                    if min_error > float(output[idx].split("\t")[2].split(":")[1]):
                        min_error = float(output[idx].split("\t")[2].split(":")[1])
                        min_index = idx
                acc_result.append(min_error)
                print("The minimum is attained in round %d" % (min_index + 1))
                clf = lgb.train(use_params, dtrain, min_index + 1, valid_sets=evallist,
                                verbose_eval=False)

            else:
                clf = lgb.train(use_params, dtrain, num_round, valid_sets=evallist,
                                verbose_eval=False)
            train_pred[eval_id] = clf.predict(eval_df.drop(columns=self.label))
            test_pred += clf.predict(dtest)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result

# class ModelFitter:
#     def __init__(self, default_dict, search_config):
#         """
#
#         :param default_dict:
#         :param search_config:
#         """
#         self.default_dict = default_dict
#         self.search_config = search_config
#         self.optimal_parameter = dict()
#         self.current_parameter = dict()
#
#     def train(self):
#         raise NotImplementedError()
#
#     def eval(self):
#         raise NotImplementedError()
#
#     def search(self):
#         """
#
#         :return:
#         """
#         self.current_parameter = deepcopy(self.default_dict)
#
#         for search_stage in self.search_config:
#             for k, v in self.optimal_parameter.items():
#                 self.current_parameter[k] = v
#             keys = sorted(search_stage)
#             possible_values = list(itertools.product(*[search_stage[key] for key in keys]))
#             best_score = -math.inf
#             for i in range(len(possible_values)):
#                 current_best_config = dict()
#                 for j in range(len(keys)):
#                     if j not in self.optimal_parameter.keys():
#                         self.current_parameter[keys[j]] = possible_values[i][j]
#                 self.train()
#                 score = self.eval()
#                 if score > best_score:
#                     best_score = score
#                     for j in range(len(keys)):
#                         current_best_config[keys[j]] = possible_values[i][j]
#                 for k, v in current_best_config.items():
#
