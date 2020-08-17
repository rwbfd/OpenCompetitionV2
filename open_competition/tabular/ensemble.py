# coding = 'utf-8'
from sklearn.model_selection import KFold
import numpy as np
from hyperopt import fmin, tpe, hp
from sklearn.ensemble import ExtraTreesClassifier
from .variable_selector import get_eval_index

max_evals = 100
extrees_default = {'n_estimator': hp.choice('n_estimator', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]),
                   'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_depth': hp.choice('max_depth', [2, 3, 4, 5, 6, 7, 8, 9])}

svm_default = {}

lf_default = {}

knn_default = {}


class EnsClassifier(object):
    def __init__(self, nclass=2):
        self.nclass = nclass

    def fit_nfold(self, df, nfold, models):
        """
        :param models: a list of models to provide training and fitting functionality.
        :return: the fitted probabilities for discrete classes; each column corresponds to the probability of each class
        """
        kfold = KFold(n_splits=nfold)
        ncol = self.nclass * len(models)
        result = np.empty(shape=(0, ncol))
        for train_index, test_index in kfold.split(df):
            train = df.loc[train_index]
            test = df.loc[test_index]
            prediction = np.empty(test.shape[0])
            for model in models:
                model.train(train)
                prediction = np.concatenate((prediction, model.predict(test)), axis=1)
            result = np.concatenate((result, prediction))
        return result

    def fit(self, train, test, models):
        """
        This is a function to fit the evaluation/test datasets for the ensemble learning techniques.
        """

        prediction = np.empty(test.shape[0])
        for model in models:
            model.train(train)
            prediction = np.concatenate((prediction, model.predict(test)), axis=1)
        return prediction

    def stack(self, predictions, y, kfold, level=2, learner=None, loss=None):
        """
        In this function, we use cross validation for find the optimal parameters for ExtraTrees, Logistic, SVM and KNN (can be configured);
        In each level of stacking, we use cross validation to find the optimal parameters for each learner;
        The final level is done by logistic regression.
        """
        if learner:
            learner_to_use = learner
        else:
            learner_to_use = ['extra_trees', 'svm', 'lr', 'knn']
        if loss:
            loss_to_use = loss
        else:
            loss_to_use = 'acc'

    # def _fit_svm(self, predictions, y, kfold):
    #
    #     def fit_svm_impl():
    #         pass

    def _fit_extra_trees(self, predictions, y, kfold, loss, space=None):
        df_folds = self._get_kfold(predictions, y, kfold)
        if space:
            search_space = space
        else:
            search_space = extrees_default

        def fit_extra_tress_impl(config):
            eval = list()
            for train_pred, test_pred, y_train, y_test in df_folds:
                clf = ExtraTreesClassifier(**config)
                clf.fit(train_pred, y_train)
                if loss == 'auc':
                    y_pred = clf.predict_proba(test_pred)[:, 0]
                else:
                    y_pred = clf.predict(test_pred)
                eval.append(-get_eval_index(y_test, y_pred, loss))
            return np.mean(eval)

        best = fmin(fn=fit_extra_tress_impl,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=max_evals)
        return best

    def _get_kfold(self, predictions, y, kfold):
        result = list()

        for train_index, test_index in kfold.split(predictions):
            train_pred, test_pred = predictions[train_index, :], predictions[train_index, :]
            y_train, y_test = y[train_index], y[test_index]
            result.append((train_pred, test_pred, y_train, y_test))
        return result

    def _rm_last_col(self, predictions):
        index = [x for x in range(predictions.shape[1]) if np.mod(x, self.nclass) != 0]
        return predictions[:, index]
