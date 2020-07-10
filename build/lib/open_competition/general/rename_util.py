# coding = 'utf_8'
from collections import OrderedDict
import pandas as pd


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
