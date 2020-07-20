# coding = 'utf-8'
import copy
import pandas as pd
from collections import OrderedDict

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def add_or_append_dict(input_dict, key, value):
    result_dict = copy.deepcopy(input_dict)
    if key in result_dict.keys():
        result_dict[key].append(value)
    else:
        result_dict[key] = [value]
    return result_dict


def get_dict_by_value(input_dict, func):
    """
    Return the keys the value of which satisfy the func.
    :param input_dict:
    :param func:
    :return:
    """
    result = list()
    for k, v in input_dict.items():
        if func(v):
            result.append(k)
    return result


def get_dict_by_value_kv(input_dict, key, value):
    """
    Example: If input_dict = {'a':{'b','c'}}, key = 'b', value = 'c', then the result is ['a'].
    :param input_dict:
    :param key:
    :param value:
    :return:
    """
    func = lambda x: x[key] == value
    return get_dict_by_value(input_dict, func)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_continuous_discrete_prefix(text):
    result = text
    to_removes = ['continuous_', 'discrete_']
    for to_remove in to_removes:
        result = remove_prefix(result, to_remove)
    return result


def get_continuous_discrete_rename_dict(original_name, continuous_vars, discrete_vars):
    result = OrderedDict()
    for name in original_name:
        if name in continuous_vars:
            result[name] = 'continuous_' + name
        elif name in discrete_vars:
            result[name] = 'discrete_' + name
        else:
            result[name] = name
    return result


def rename_continuous_discrete(csv_to_rename, csv_name_after_rename, continuous_vars_csv, discrete_vars_csv,
                               column_names='column_names'):
    df_to_rename = pd.read_csv(csv_to_rename, engine='python')
    continuous_vars_df = pd.read_csv(continuous_vars_csv, engine='python')
    discrete_vars_df = pd.read_csv(discrete_vars_csv, engine='python')
    continuous_vars_list = continuous_vars_df[column_names].tolist()
    discrete_vars_list = discrete_vars_df[column_names].tolist()

    rename_dict = get_continuous_discrete_rename_dict(df_to_rename.columns, continuous_vars_list, discrete_vars_list)
    df_to_rename = df_to_rename.rename(columns=rename_dict)
    df_to_rename.to_csv(csv_name_after_rename, index=False)
    return


def split_df(df, n_splits, shuffle=True):
    """
    Split the dataframe into n folds; Each contains a list of tuples
    :param df: The dataset to be splitted
    :param n_splits: The number of folds
    :param shuffle: Whether to shuffle the data; default is "True"
    :return:
    """
    df_copy = df.copy(deep=True)
    splitter = KFold(n_splits=n_splits, shuffle=shuffle)
    result = list()

    for train_index, test_index in splitter.split(df_copy):
        result.append((df_copy.iloc[train_index], df_copy.iloc[test_index]))

    return result


def get_var_type(df):
    columns = df.columns
    continuous_vars = [x for x in columns if x.startswith('continuous_')]
    discrete_vars = [x for x in columns if x.startswith('discrete_')]
    other_vars = list()
    for column in columns:
        if column not in continuous_vars and column not in discrete_vars:
            other_vars.append(column)
    return {'continuous': continuous_vars,
            'discrete': discrete_vars,
            'other': other_vars}


def get_cont_var(df):
    var_types = get_var_type(df)
    return var_types['continuous']


def get_dis_var(df):
    var_types = get_var_type(df)
    return var_types['discrete']


def get_var_end_with(df, ends):
    columns = df.columns
    return [x for x in columns if x.endswith(ends)]


def eval_classification(y_true, y_pred_class):
    result = OrderedDict()
    result['acc'] = accuracy_score(y_true, y_pred_class)
    result['recall'] = recall_score(y_true, y_pred_class)
    result['precision'] = precision_score(y_true, y_pred_class)
    result['macro_f1'] = f1_score(y_true, y_pred_class, average='macro')
    result['micro_f1'] = f1_score(y_true, y_pred_class, average='micro')
    result['confusion_matrix'] = confusion_matrix(y_true, y_pred_class)
    return result


def eval_binary_classification(y_true, y_pred_class, y_pred_prob):
    result = eval_classification(y_true, y_pred_class)
    result['auc'] = roc_auc_score(y_true, y_pred_prob)
    return result
