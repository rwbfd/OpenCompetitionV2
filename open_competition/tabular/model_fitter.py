# coding = 'utf-8'
from copy import deepcopy
import itertools
import math
from dataclasses import dataclass, asdict
import multiprocessing
from hyperopt import fmin, tpe, hp
import hyperopt.pyll
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
cpu_count = multiprocessing.cpu_count()


@dataclass
class XGBOpt:
    nthread: hyperopt.pyll.base.Apply = hp.choice('nthread', [cpu_count])
    eval_metric: hyperopt.pyll.base.Apply = hp.choice('eval_metric', ['error'])
    objective: hyperopt.pyll.base.Apply = hp.choice('objective', ['binary:logistic'])
    max_depth: hyperopt.pyll.base.Apply = hp.choice('max_depth', [4, 5, 6, 7, 8])
    early_stopping_rounds: hyperopt.pyll.base.Apply = hp.choice('early_stopping_rounds', [50])
    num_round: hyperopt.pyll.base.Apply = hp.choice('num_round', [1000])
    eta: hyperopt.pyll.base.Apply = hp.uniform('eta', 0.01, 0.1)
    subsample: hyperopt.pyll.base.Apply = hp.uniform('subsample', 0.8, 1)
    colsample_bytree: hyperopt.pyll.base.Apply = hp.uniform('colsample_bytree', 0.3, 1)
    gamma: hyperopt.pyll.base.Apply = hp.choice('gamma', [0, 1, 5])

@dataclass
class LGBOpt:


class FitterBase(object):
    def __init__(self, label, metric):
        self.label = label
        self.metric = metric
        self.opt_params = dict()

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


class XgBoostFitter(FitterBase):
    def __init__(self,  label='label', metric='error', opt: XGBOpt = None, max_eval=100):
        super(XgBoostFitter, self).__init__(label, metric)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = XGBOpt()
        self.clf = None
        self.max_eval = max_eval

    def train(self, train_df, eval_df, params=None, use_early_stop=True, verbose_eval=False):
        dtrain = xgb.DMatrix(train_df.drop(columns=[self.label]), train_df[self.label])
        deval = xgb.DMatrix(eval_df.drop(columns=[self.label]), eval_df[self.label])
        evallist = [(deval, 'eval')]
        if params is None:
            use_params = self.opt_params
        else:
            use_params = params
        if use_early_stop:
            self.clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                 early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=verbose_eval)
        else:
            self.clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                 verbose_eval=verbose_eval)

    def search(self, train_df, eval_df):
        self.opt_params = dict()
        deval = xgb.DMatrix(eval_df.drop(columns=[self.label]))

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(deval)
            else:
                y_pred = (self.clf.predict(deval) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.spilt(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params)
                deval = xgb.DMatrix(eval_df.drop(columns=[self.label]))
                if self.metric == 'auc':
                    y_pred = self.clf.predict(deval)
                else:
                    y_pred = (self.clf.predict(deval) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def get_rand_param(self):
        return hyperopt.pyll.stochastic.sample(asdict(self.opt))

    def train_k_fold(self, k_fold, data, column_name='xgb_predict', params=None, use_early_stop=True,
                     verbose_eval=False):
        data_copy = data.copy(deep=True)
        if params is not None:
            use_params = params
        else:
            use_params = self.opt_params
        data_copy[column_name] = np.NaN

        for train_id, eval_id in k_fold.split(data_copy):
            train_df = data_copy.loc[train_id]
            eval_df = data_copy.loc[eval_id]
            dtrain = xgb.DMatrix(train_df.drop(columns=[self.label]), train_df[self.label])
            deval = xgb.DMatrix(eval_df.drop(columns=[self.label]), eval_df[self.label])
            evallist = [(deval, 'eval')]
            if use_early_stop:
                clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'],
                                early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=verbose_eval)
            else:
                clf = xgb.train(use_params, dtrain, num_boost_round=params['num_round'], evals=evallist,
                                verbose_eval=verbose_eval)
            y_pred = clf.predict(deval)
            data_copy.loc[eval_id, column_name] = y_pred
        return data_copy[column_name]


class LGBFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: XGBOpt = None, max_eval=100):

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
#                     self.optimal_parameter[k] = v