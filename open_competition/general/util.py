# coding = 'utf-8'
import copy


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