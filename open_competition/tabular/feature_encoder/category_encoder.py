# coding = 'utf-8'

import category_encoders as ce
from ...general.string_util import remove_continuous_discrete_prefix


class CategoryEncoder(object):
    def __init__(self):
        self.result_list = list()

    def fit(self, df, y, targets, configurations):
        for target in targets:
            for config in configurations:
                self._fit_one(df, y, target, config)

    def _fit_one(self, df, y, target, config):
        method, parameter = config[0], config[1]
        if method == 'woe':
            self._fit_woe(df, y, target, config)

    def _fit_woe(self, df, y, target, config):
        woe_encoder = ce.woe.WOEEncoder(cols=target)
        woe_encoder.fit(df[target], df[y])
        name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_woe"
        self.result_list.append(('woe', name, target, woe_encoder))

    def transform(self, df, y=None):
        result_df = df.copy(deep=True)
        for method, name, target, encoder in self.result_list:
            if method == 'woe':
                if y:
                    result_df[name] = encoder.transform(df[target], df[y])
                else:
                    result_df[name] = encoder.transform(df[target])
        return result_df

