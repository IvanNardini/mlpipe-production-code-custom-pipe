'''
data_preprocessing module contains machine learning object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Preprocessing:

    def data_preparer(self, data, dropped_columns, renamed_columns):
        '''
        Drop and Rename columns
        :params: data, columns_to_drop
        :return: DataFrame
        '''
        data = data.copy()
        data.drop(dropped_columns, axis=1, inplace=True)
        data.rename(columns=renamed_columns, inplace=True)
        return data

    def missing_imputer(self, data, missing_predictors, replace='missing'):
        '''
        Imputes '?' character with 'missing' label
        :params: data, missing_predictors, replace
        :return: Series
        '''
        data = data.copy()
        for var in missing_predictors:
            data[var].replace('?', replace)
        return data
    
    def binner(self, data, binning_meta):
        '''
        Create bins based on variable distributions
        :params: data, var, new_var_name, bins, bins_labels
        :return: Series
        '''
        data = data.copy()
        for var, meta in binning_meta.items():
            data[meta['var_name']] = pd.cut(data[var], bins = meta['bins'], labels=meta['bins_labels'], include_lowest= True)
            data.drop(var, axis=1, inplace=True)
        return data

    def encoder(self, data, encoding_meta):
        '''
        Encode all variables for training
        :params: data, var, mapping
        :return: DataFrame
        '''
        data = data.copy()
        for var, meta in encoding_meta.items():
            if var not in data.columns.values.tolist():
                pass
            data[var] = data[var].map(encoding_meta)
        return data


    
    
