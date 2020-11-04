from copy import deepcopy
from dataclasses import dataclass, asdict

import cudf
import hyperopt.pyll
import numpy as np
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
from cuml.metrics import roc_auc_score
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVC
from hyperopt import fmin, tpe, hp


@dataclass
class LROpt:
    penalty: any = hp.choice('penalty', ['l2', 'l1', 'none'])
    # penalty: any = hp.choice('penalty', ['l2'])
    C: any = hp.uniform('C', 0.5, 10)
    max_iter: any = hp.choice('max_iter', [1000])
    solver: any = hp.choice('solver', ['qn'])
    linesearch_max_iter: any = hp.choice('linesearch_max_iter', [50, 75, 100])
    output_type: any = hp.choice('output_type', ['numpy'])

    @staticmethod
    def get_common_params():
        return {'penalty': 'l2', 'C': 0.5}


@dataclass
class KNNOpt:
    n_neighbors: any = hp.choice('n_neighbors', [2, 5, 10])
    weights: any = hp.choice('weights', ['uniform'])
    metric: any = hp.choice('metric', ['euclidean'])
    algorithm: any = hp.choice('algorithm', ['brute'])

    @staticmethod
    def get_common_params():
        return {'n_neighbors': 5, 'weights': 'uniform'}


@dataclass
class SVMOpt:
    C: any = hp.uniform('C', 0, 10)
    kernel: any = hp.choice('kernel', ['rbf', 'sigmoid'])
    gamma: any = hp.choice('gamma', ['scale', 'auto'])
    probability: any = hp.choice('probability', [True])

    @staticmethod
    def get_common_params():
        return {"C": 1, 'kernel': 'rbf', 'probability': True}


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
        elif self.metric == 'auc':  # TODO: Add a warning checking if y_predict is all [0, 1], it should be probability
            return 1 - roc_auc_score(y, y_pred)
        else:
            raise Exception("Not implemented yet.")

    def get_rand_param(self):
        if self.opt is None:
            return None
        else:
            return hyperopt.pyll.stochastic.sample(asdict(self.opt))


class CumlLRFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: LROpt = None, max_eval=10):
        super(CumlLRFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = LROpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        train_df, eval_df = cudf.DataFrame(train_df), cudf.DataFrame(eval_df)
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = LogisticRegression(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.iloc[train_id, :]
                eval_df = data.iloc[eval_id, :]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = cudf.Series(np.empty(train_data.shape[0]))
        test_pred = cudf.Series(np.empty(test_data.shape[0]))
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id, :]
            eval_df = train_data.iloc[eval_id, :]
            self.train(train_df, eval_df, params)
            train_pred[eval_id] = self.clf.predict_proba(eval_df.drop(columns=self.label)).iloc[:, 1].values
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict_proba(dtest).iloc[:, 1]

        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result


class CumlKNNFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: KNNOpt = None, max_eval=10):
        super(CumlKNNFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = KNNOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        train_df, eval_df = cudf.DataFrame(train_df), cudf.DataFrame(eval_df)
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],

        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = KNeighborsClassifier(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.iloc[train_id, :]
                eval_df = data.iloc[eval_id, :]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id]
            eval_df = train_data.iloc[eval_id]
            self.train(train_df, eval_df, params)
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label))
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(dtest)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result


class CumlSVMFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: SVMOpt = None, max_eval=100):
        super(CumlSVMFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = SVMOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        train_df, eval_df = cudf.DataFrame(train_df), cudf.DataFrame(eval_df)
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = SVC(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.iloc[train_id, :]
                eval_df = data.iloc[eval_id, :]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = cudf.Series(np.empty(train_data.shape[0]))
        test_pred = cudf.Series(np.empty(test_data.shape[0]))
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id, :]
            eval_df = train_data.iloc[eval_id, :]
            self.train(train_df, eval_df, params)
            train_pred[eval_id] = self.clf.predict_proba(eval_df.drop(columns=self.label)).iloc[:, 1].values
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict_proba(dtest).iloc[:, 1]
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result
