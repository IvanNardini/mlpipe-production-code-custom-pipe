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

    def Data_Preparer(self, data, dropped_columns, renamed_columns):
        '''
        Drop and Rename columns
        :params: data, columns_to_drop
        :return: DataFrame
        '''
        data = data.copy()
        data.drop(dropped_columns, axis=1, inplace=True)
        data.rename(columns=renamed_columns, inplace=True)
        return data

    def Missing_Imputer(self, data, missing_predictors, replace='missing'):
        '''
        Imputes '?' character with 'missing' label
        :params: data, missing_predictors, replace
        :return: Series
        '''
        data = data.copy()
        for var in missing_predictors:
            data[var] = data[var].replace('?', replace)
        return data

    def Binner(self, data, binning_meta):
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

    def Encoder(self, data, encoding_meta):
        '''
        Encode all variables for training
        :params: data, var, mapping
        :return: DataFrame
        '''
        data = data.copy()
        for var, meta in encoding_meta.items():
            if var not in data.columns.values.tolist():
                pass
            data[var] = data[var].map(meta)
        return data

    def Dumminizer(self, data, columns_to_dummies, dummies_meta):
        '''
        Generate dummies for nominal variables
        :params: data, columns_to_dummies, dummies_meta
        :return: DataFrame
        '''
        data = data.copy()
        for var in columns_to_dummies:
            cat_names = sorted(dummies_meta[var])
            obs_cat_names = sorted(list(set(data[var].unique())))
            dummies = pd.get_dummies(data[var], prefix=var)
            data = pd.concat([data, dummies], axis=1)
            if obs_cat_names != cat_names: #exception: when label misses 
                cat_miss_labels = ["_".join([var, cat]) for cat in cat_names if cat not in obs_cat_names] #syntetic dummy
                for cat in cat_miss_labels:
                    data[cat] = 0 
            data = data.drop(var, 1)
        return data

    def Scaler(self, data, columns_to_scale):
        '''
        Scale variables
        :params:  data, columns_to_scale
        :return: DataFrame
        '''
        data = data.copy()
        scaler = MinMaxScaler()
        scaler.fit(data[columns_to_scale])
        data[columns_to_scale] = scaler.transform(data[columns_to_scale])
        return data

   
            

    
    
