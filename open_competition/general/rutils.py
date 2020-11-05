# coding = 'utf-8'
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter


def pd_to_r(df):
    """
    :param df: pandas data frame
    :return r dataframe object
    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(df)
    return r_from_pd_df

def r_to_pd(df):
    """
    :param df: r dataframe object
    :return pandas dataf frame
    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(df)
    return pd_from_r_df
