# coding = 'utf-8'


class GroupByEncoder(object):
    def __init__(self, smoothing_coefficients=None):
        if not smoothing_coefficients:
            self.smoothing_coefficients = [0]
        else:
            self.smoothing_coefficients = smoothing_coefficients

    def encode(self, df, targets, groupby_list, operations_list):
        result = df.copy(deep=True)
        for target in targets:
            for index in range(len(groupby_list)):
                groupby = groupby_list[index]
                operations = operations_list[index]
                for operation in operations:
                    groupby_result = self._encode_one(result, target, groupby, operation)
                    name = target + '_groupby_' + '_'.join(groupby) + '_op_' + operation
                    groupby_result = groupby_result.rename(columns={target: name})
                    result = result.merge(groupby_result, on=groupby, how='left')

        return result

    def _encode_one(self, df, target, groupby_vars, operation):
        result = df.groupby(groupby_vars, as_index=False).agg({target: operation})
        return result
