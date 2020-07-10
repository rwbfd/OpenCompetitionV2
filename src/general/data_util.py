# encoding = 'utf-8'
from sklearn.model_selection import KFold


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


def find_continuous_discrete_variables(df):
    columns = df.columns
    continuous_vars = [x for x in columns if x.startwith('continuous_')]
    discrete_vars = [x for x in columns if x.startwith('discrete_')]
    other_vars = list()
    for column in columns:
        if column not in continuous_vars and column not in discrete_vars:
            other_vars.append(column)
    return {'continuous': continuous_vars,
            'discrete': discrete_vars,
            'other': other_vars}
