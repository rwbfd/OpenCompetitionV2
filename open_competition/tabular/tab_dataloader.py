# coding = 'utf-8'
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset

class TabDataLoader(Dataset):
    """
    This is a data loader class designed for tabular data.
    The tabular data will be fed into an entity embedding layer, an entity embedding with vicinity information and a direct densely-connected layer.

    """

    def __init__(self, data_pd, tab_dataloader_opt: TabDataOpt):
        self.data_pd = data_pd.copy(deep=True)
        self.tab_dataloader_opt = tab_dataloader_opt
        self.use_dis_var = False
        self.use_vin_var = False
        self.use_conti_var = False
        self.all_vars = list()
        if self.tab_dataloader_opt.dis_vars_entity is not None:
            self.all_vars += self.tab_dataloader_opt.dis_vars_entity
            self.use_dis_var = True
        if self.tab_dataloader_opt.dis_vars_vic is not None:
            self.all_vars += self.tab_dataloader_opt.dis_vars_vic
            self.use_vin_var = True

        if self.tab_dataloader_opt.conti_vars is not None:
            self.all_vars += self.tab_dataloader_opt.conti_vars
            self.use_conti_var = True
        self.all_vars.append(self.tab_dataloader_opt.label)
        self.length = data_pd.shape[0]

        if not (set(self.all_vars) <= set(data_pd.columns)):
            logging.error("""
            The options contains variables that are NOT in the actual dataframe. 
            """)
        self.shape = self.data_pd.shape[0]

        def is_invalid(x):
            if pd.isnull(x):
                return True

            if x is np.inf or x is -np.inf:
                return True

            return False

        for var in self.all_vars:
            if self.data_pd[var].map(is_invalid).any():
                logging.error("""
                Variable %s contains NaN or inf values. Please impute the values first.
                """ % var
                              )
        for var in self.tab_dataloader_opt.dis_vars_entity:
            unique_val = self.data_pd[var].unique()
            index = range(len(unique_val))
            map_dict = dict(zip(unique_val, index))

            self.data_pd.loc[:, var] = self.data_pd.loc[:, var].map(lambda x: map_dict[x])

    def __getitem__(self, item):
        line = self.data_pd.iloc[item, :]

        result = dict()

        for var in self.all_vars:
            result[var] = line[var]

        return result

    def __len__(self):
        return self.length