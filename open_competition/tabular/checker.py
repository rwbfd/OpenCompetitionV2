# coding = 'utf-8'
from collections import OrderedDict
from ..general.util import add_or_append_dict


class DataChecker:
    def __init__(self):
        self.problem_result_dict = OrderedDict()

    def check(self, df, reset=True):
        if reset:
            self.problem_result_dict = OrderedDict()
        for column in df.columns:
            self.check_unique_value(df, column)
        return self.problem_result_dict

    def check_unique_value(self, df, column):
        if len(df[column].unique()) == 1:
            add_or_append_dict(self.problem_result_dict, column, {'unique_value'})
