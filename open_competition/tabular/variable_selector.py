# coding = 'utf-8'
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import ray
import numpy as np
from ind_cols import get_ind_col
from .model_fitter import LGBFitter, XGBFitter


def permutate_selector(train_df, eval_df, y, variables=None, metric='acc', **kwargs):  # TODO Add more metric

    """
    Return the importance of variables based on permutation loss

    :param train_df: training data set
    :param eval_df: eval data set
    :param y: name of the target variable
    :param variables: the variables to select perform the select; if None, then all the variables except target variable will be selected
    :param metric: the metric to determine the order; higher value indicate better performance
    :param **kwargs: argument for logistic regression

    returns: result after permutation
    """

    @ray.remote()
    def fit_and_predict(train_df, eval_df, y, variables, metric, start=None, **kwargs):
        if start is None:
            clf = LogisticRegression(**kwargs)
        else:
            clf = LogisticRegression(warm_start=start, **kwargs)
        clf.fit(train_df[variables], train_df[y])
        y_pred = clf.predict(eval_df[variables])

        if metric == 'acc':  # TODO Add more metric
            score = accuracy_score(eval_df[y], y_pred)
        else:
            score = None
        return score, clf.coef_

    @ray.remote()
    def fit_permute_and_predict(train_df, eval_df, y, variables, metric, start, permute_var, **kwargs):
        train_df[permute_var] = np.random.permutation(train_df[permute_var])
        score, _ = fit_and_predict(train_df, eval_df, y, variables, metric, start, **kwargs)
        return (permute_var, score)

    ray.init()

    ind_col = get_ind_col(train_df)
    if variables is not None:
        var_to_use = [x for x in ind_col if x in variables]
    else:
        var_to_use = [x for x in ind_col if x != y]

    result_dict = dict()

    score, warm_start = fit_and_predict(train_df, eval_df, y, var_to_use, metric, None, **kwargs)
    result_dict['origin'] = score

    train_df_id = ray.put(train_df)
    eval_df_id = ray.put(eval_df)

    var_to_use_id = ray.put(var_to_use)
    start_id = ray.put(warm_start)
    result = [
        fit_permute_and_predict.remote(train_df_id, eval_df_id, y, var_to_use_id, start_id, permute_var, **kwargs, ) for
        permute_var in var_to_use]
    result_list = ray.get(result)

    for var, score in result_list:
        result_dict[var] = score
    ray.shutdown()
    return result_dict


def tree_selector(train_df, eval_df, y, opt, variables=None, type='lgb'):
    """
    This function select variable importance using built functions from xgboost or lightgbm
    """

    def lgb_selector():
        pass

    def xgb_selector():
        pass

    if type == 'lgb':
        pass


def shap_selector(train_df, eval_df, y, opt, variables=None, type='lgb'):
    """
    This returns the shap explainer so that one can use it for variable selection.
    The base tree model we use will select the best iterations
    :param train_df: training dataset
    :param eval_df: eval dataset
    :param y: the target variable name
    :param opt: training argument for boosting parameters
    :param variables: variables to make the selection. If none, will use all the selector
    :param type; 'lgb' or 'xgb'. The tree used for computing shap values.

    returns: shap explainer
    """
    pass


def vif_selector(data_df, y, variables=None):
    pass
