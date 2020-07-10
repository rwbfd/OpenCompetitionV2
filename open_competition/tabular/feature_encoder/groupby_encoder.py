# coding = 'utf-8'


class GroupbyEncoder(object):
    def __init__(self):
        self.groupby_result_list = list()

    def fit(self, df, targets, groupby_op_list):
        self.groupby_result_list = list()
        for target in targets:
            for groupby, operations in groupby_op_list:
                for operation in operations:
                    groupby_result = self._fit_one(df, target, groupby, operation)
                    name = target + '_groupby_' + '_'.join(groupby) + '_op_' + operation
                    groupby_result = groupby_result.rename(columns={target: name})
                    self.groupby_result_list.append((groupby, groupby_result))

    def transform(self, df):
        result = df.copy(deep=True)
        for groupby, groupby_result in self.groupby_result_list:
            result = result.merge(groupby_result, on=groupby, how='left')
        return result

    def _fit_one(self, df, target, groupby_vars, operation):
        result = df.groupby(groupby_vars, as_index=False).agg({target: operation})
        return result
