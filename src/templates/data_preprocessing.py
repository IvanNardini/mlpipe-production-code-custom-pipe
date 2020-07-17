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

    def binner(self, data, var, new_var_name, bins, bins_labels):
        '''
        Create bins based on variable distributions
        :params: data, var, new_var_name, bins, bins_labels
        :return: Series
        '''
        data = data.copy()


        data[new_var_name] = pd.cut(data[var], bins = bins, labels=bins_labels, include_lowest= True)
        data.drop(var, axis=1, inplace=True)
        return data[new_var_name]

    
    
