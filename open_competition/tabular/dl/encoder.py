# coding = 'utf-8'
import numpy as np
import pandas as pd
import tqdm

def encode_label(x):
    unique=sorted(list(set([str(item) for item in np.unique(x)])))
    kv = {unique[i]: i for i in range(len(unique))}
    vfunc = np.vectorize(lambda x: kv[str(x)])
    return vfunc(x)

def encode_label_mat(x):
    _, ncol = x.shape
    result = np.empty_like(x, dtype=int)
    for col in range(ncol):
        result[:,col] = encode_label(x[:, col])
    return result

def impute_nan(x, method='median'):
    _, ncol = x.shape
    result = np.empty_like(x)

    for col in range(ncol):
        if method == 'median':
            data = x[:, col]
            impute_value = np.median(data[~pd.isnull(data) & (data != np.inf) & (data != -np.inf)])
        else:
            raise NotImplementedError()

        func = np.vectorize(lambda x: impute_value if pd.isnull(x) else x)
        result[:, col] = func(x[:, col])
    return result


def get_uniform_interval(minimum, maximum, nbins):
    result = [minimum]
    step_size = (float(maximum - minimum)) / nbins
    for index in range(nbins - 1):
        result.append(minimum + step_size * (index + 1))
    result.append(maximum)
    return result


def get_interval_v2(x, sorted_intervals):
    if pd.isnull(x):
        return -1
    if x == np.inf:
        return -2
    if x == -np.inf:
        return -3
    interval = 0
    found = False
    sorted_intervals.append(np.inf)
    while not found and interval < len(sorted_intervals) - 1:
        if sorted_intervals[interval] <= x < sorted_intervals[interval + 1]:
            return interval
        else:
            interval += 1


def get_quantile_interval(data, nbins):
    quantiles = get_uniform_interval(0, 1, nbins)
    return list(np.quantile(data[(~pd.isnull(data)) & (data != np.inf) & (data != -np.inf)], quantiles))


def discretize(x, nbins=20):
    nrow, ncol = x.shape
    result = np.empty_like(x)
    interval_list = list()
    for col in range(ncol):
        intervals = sorted(list(set(get_quantile_interval(x[:, col], nbins))))
        interval_centroid = list()

        for i in range(len(intervals) - 1):
            interval_centroid.append(0.5 * (intervals[i] + intervals[i + 1]))
        func = np.vectorize(lambda x: get_interval_v2(x, intervals))
        result[:, col] = encode_label(func(x[:, col]))
        interval_list.append(interval_centroid)
    return result.astype(np.int64), interval_list