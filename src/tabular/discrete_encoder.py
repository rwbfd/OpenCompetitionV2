# coding = 'utf-8'

class DiscreteEncoder:
    def __init__(self):
        pass

    def encode(self, df, targets, configurations):
        result = df.copy(deep=True)
        for target in targets:
            for method, nbins in configurations:
                result = self._encode_one(result, target, method, nbins)
        return result

    def _encode_one(self, df, target, method, nbins):
        result = df
        if method == 'uniform':
            intervals = self._get_uniform_intervals(df, target, nbins)
            name = target + "_uniform_" + str(nbins)
            result[name] = result[target].map(lambda x: get_interval(x, intervals))
        elif method == 'quantile':
            intervals = self._get_quantile_intervals(df, target, nbins)
            name = target + "_quantile_" + str(nbins)
            result[name] = result[target].map(lambda x: get_interval(x, intervals))
        else:
            raise Exception("Not Implemented Yet")
        return result

    def _get_uniform_intervals(self, df, target, nbins):
        target_var = df[target]
        minimum = target_var.min()
        maximum = target_var.max()

        intervals = get_uniform_interval(minimum, maximum, nbins)
        return intervals

    def _get_quantile_intervals(self, df, target, nbins):
        return get_quantile_interval(df[target], nbins)


def get_interval(x, sorted_intervals):
    interval = 0
    found = False
    while not found and interval < len(sorted_intervals) - 1:
        if sorted_intervals[interval] <= x <= sorted_intervals[interval + 1]:
            found = True
            return "i_" + str(interval)
        else:
            interval += 1


def get_uniform_interval(minimum, maximum, nbins):
    result = [minimum]
    step_size = (float(maximum - minimum)) / nbins
    for index in range(nbins - 1):
        result.append(step_size * (index + 1))
    result.append(maximum)
    return result


def get_quantile_interval(data, nbins):
    quantiles = get_uniform_interval(0, 1, nbins)
    print(quantiles)
    return list(data.quantile(quantiles))
