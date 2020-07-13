# coding = 'utf-8'
from collections import OrderedDict
from copy import deepcopy
import itertools
import math


class ModelFitter:
    def __init__(self, default_dict, search_config):
        """

        :param default_dict:
        :param search_config:
        """
        self.default_dict = default_dict
        self.search_config = search_config
        self.optimal_parameter = dict()
        self.current_parameter = dict()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def search(self):
        self.current_parameter = deepcopy(self.default_dict)

        for search_stage in self.search_config:
            for k, v in self.optimal_parameter.items():
                self.current_parameter[k] = v
            keys = sorted(search_stage)
            possible_values = list(itertools.product(*[search_stage[key] for key in keys]))
            best_score = -math.inf
            for i in range(len(possible_values)):
                current_best_config = dict()
                for j in range(len(keys)):
                    if j not in self.optimal_parameter.keys():
                        self.current_parameter[keys[j]] = possible_values[i][j]
                self.train()
                score = self.eval()
                if score > best_score:
                    best_score = score
                    for j in range(len(keys)):
                        current_best_config[keys[j]] = possible_values[i][j]
                for k, v in current_best_config.items():
                    self.optimal_parameter[k] = v
